"""
Launch training from a single YAML config.

Ported from JOLMo (https://github.com/jakespringer/JOLMo), keeping the "normal" and "sam"
train module types.

Usage::

    torchrun --nproc-per-node=1 src/scripts/launch_from_yaml.py CONFIG.yaml [dot.notation=overrides ...]
    python src/scripts/launch_from_yaml.py CONFIG.yaml --dry-run
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from olmo_core.config import Config
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
)
from olmo_core.distributed.utils import get_rank
from olmo_core.io import is_url
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.common import Duration
from olmo_core.train.train_module import TrainModule, TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    SAMScheduler,
    TransformerSAMConfig,
    TransformerSAMTrainModule,
)
from olmo_core.train.trainer import Trainer
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


@dataclass
class WandBSettings(Config):
    """
    W&B settings. A :class:`~olmo_core.train.callbacks.WandBCallback` is added (or an existing
    one updated) when ``enabled`` is ``True``.
    """

    enabled: bool = False
    name: Optional[str] = None
    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


@dataclass
class YamlExperimentConfig(Config):
    """
    Top-level YAML schema for :func:`main`.
    """

    # Either provide full model config or a factory + args
    model: Optional[TransformerConfig] = None
    model_factory: Optional[str] = None
    model_factory_args: Optional[Dict[str, Any]] = None

    dataset: Optional[NumpyDatasetConfig] = None
    data_loader: Optional[NumpyDataLoaderConfig] = None

    # "normal" or "sam"
    train_module_type: str = "normal"
    train_module: Optional[TransformerTrainModuleConfig] = None
    # SAM-only config (optional)
    sam: Optional[TransformerSAMConfig] = None
    sam_scheduler: Optional[SAMScheduler] = None  # schedule for sam rho, if provided

    trainer: Optional[TrainerConfig] = None
    wandb: Optional[WandBSettings] = None
    # Optional mapping of validation dataset label -> list of paths. If provided,
    # an LM evaluator will be added that reports metrics per label.
    validation_datasets: Optional[Dict[str, List[str]]] = None
    # Optional eval interval for validation datasets. Defaults to 1000 if not set.
    validation_eval_interval: Optional[int] = None
    # Optional explicit list of training steps at which to run validation evals,
    # in addition to ``validation_eval_interval``.
    validation_eval_steps: Optional[List[int]] = None

    init_seed: int = 12536
    load_path: Optional[str] = None
    load_optim_state: Optional[bool] = None
    load_trainer_state: Optional[bool] = None
    n_tokens: Optional[int] = None


def _build_model(cfg: YamlExperimentConfig) -> Transformer:
    if cfg.model is not None:
        return cfg.model.build(init_device="meta")
    if cfg.model_factory:
        factory = getattr(TransformerConfig, cfg.model_factory, None)
        if factory is None:
            raise ValueError(f"Unknown model factory: {cfg.model_factory}")
        kwargs = cfg.model_factory_args or {}
        model_conf = factory(**kwargs)
        return model_conf.build(init_device="meta")
    raise ValueError("You must provide either 'model' or 'model_factory'.")


def _build_train_module_normal(cfg: YamlExperimentConfig, model: Transformer) -> TrainModule:
    assert cfg.train_module is not None
    return cfg.train_module.build(model)


def _build_train_module_sam(cfg: YamlExperimentConfig, model: Transformer) -> TrainModule:
    import torch.distributed.checkpoint.state_dict as dist_cp_sd

    from olmo_core.config import DType

    assert cfg.train_module is not None
    kwargs = cfg.train_module.as_dict(exclude_none=True, recurse=False)
    if kwargs.pop("ema", None) is not None:
        raise ValueError("Weight EMA (`ema`) is not supported with the SAM train module.")
    if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
        kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
    if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
        kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
    if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
        kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
    return TransformerSAMTrainModule(
        model=model, sam_config=cfg.sam, sam_scheduler=cfg.sam_scheduler, **kwargs
    )


def _ensure_wandb_callback(trainer: Trainer, cfg: YamlExperimentConfig) -> None:
    settings = cfg.wandb or WandBSettings(enabled=False)
    if not settings.enabled:
        return

    # If a wandb callback already exists, augment it; otherwise add a new one.
    existing_cb: Optional[WandBCallback] = None
    for cb in trainer.callbacks.values():
        if isinstance(cb, WandBCallback):
            existing_cb = cb
            break

    if existing_cb is None:
        trainer.add_callback(
            "wandb",
            WandBCallback(
                enabled=True,
                name=settings.name,
                project=settings.project,
                entity=settings.entity,
                group=settings.group,
                tags=settings.tags,
                notes=settings.notes,
            ),
        )
    else:
        for field_name in ("name", "project", "entity", "group", "tags", "notes"):
            if (value := getattr(settings, field_name)) is not None:
                setattr(existing_cb, field_name, value)


def _ensure_validation_callbacks(trainer: Trainer, cfg: YamlExperimentConfig) -> None:
    """
    If 'validation_datasets' are provided in the YAML, attach an LM evaluator that
    evaluates each dataset individually (via per-file metadata labels).
    """
    if not cfg.validation_datasets:
        return

    assert cfg.dataset is not None
    assert cfg.train_module is not None

    # Derive sequence length and tokenizer for eval dataset from training config.
    seq_len = getattr(cfg.dataset, "sequence_length", None)
    if seq_len is None:
        seq_len = getattr(cfg.train_module, "max_sequence_length", None)
    if seq_len is None:
        raise ValueError("Cannot determine sequence length for validation datasets")

    tokenizer = getattr(cfg.dataset, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Training dataset must specify a tokenizer to run validation")

    val_paths: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for label, paths in cfg.validation_datasets.items():
        for p in paths:
            val_paths.append(p)
            metadata.append({"label": label})

    eval_dataset_cfg = NumpyPaddedFSLDatasetConfig(
        sequence_length=seq_len,
        tokenizer=tokenizer,
        work_dir=getattr(cfg.dataset, "work_dir", None),
    )
    eval_dataset_cfg.paths = val_paths
    eval_dataset_cfg.metadata = metadata

    eval_cb_cfg = LMEvaluatorCallbackConfig(
        eval_dataset=eval_dataset_cfg,
        eval_interval=cfg.validation_eval_interval or 1000,
        fixed_steps=list(cfg.validation_eval_steps) if cfg.validation_eval_steps else None,
    )
    cb = eval_cb_cfg.build(trainer)
    if cb is not None:
        trainer.add_callback("lm_eval_validation", cb)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} CONFIG.yaml [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Launch training from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path/URL to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the parsed config and exit.")
    return parser


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    return _parser().parse_known_args()


def main():
    opts, overrides = _parse_args()
    cfg = YamlExperimentConfig.from_file(opts.config, overrides=overrides)

    missing = [
        name
        for name, val in dict(
            dataset=cfg.dataset,
            data_loader=cfg.data_loader,
            train_module=cfg.train_module,
            trainer=cfg.trainer,
        ).items()
        if val is None
    ]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    assert cfg.dataset is not None
    assert cfg.data_loader is not None
    assert cfg.train_module is not None
    assert cfg.trainer is not None

    if opts.dry_run:
        if get_rank() == 0:
            print(cfg)
        return

    prepare_training_environment(shared_filesystem=not is_url(cfg.trainer.save_folder))

    seed_all(cfg.init_seed)

    model = _build_model(cfg)
    tmt = cfg.train_module_type.lower()
    if tmt == "sam":
        train_module = _build_train_module_sam(cfg, model)
    elif tmt == "normal":
        train_module = _build_train_module_normal(cfg, model)
    else:
        raise ValueError(f"Unknown train_module_type '{cfg.train_module_type}' (expected 'normal' or 'sam')")

    dataset = cfg.dataset.build()
    data_loader = cfg.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = cfg.trainer.build(train_module, data_loader)

    # Override max_duration if n_tokens is specified
    if cfg.n_tokens is not None:
        trainer.max_duration = Duration.tokens(cfg.n_tokens)

    # Save config for W&B and checkpoints.
    for callback in trainer.callbacks.values():
        if isinstance(callback, ConfigSaverCallback):
            callback.config = cfg.as_config_dict()
            break

    _ensure_wandb_callback(trainer, cfg)
    _ensure_validation_callbacks(trainer, cfg)

    # Auto-load latest checkpoint if present; else optionally init-from `load_path` (model weights only).
    loaded = False
    if not trainer.no_checkpoints:
        loaded = trainer.maybe_load_checkpoint()
    if not loaded and cfg.load_path:
        load_optim = cfg.load_optim_state if cfg.load_optim_state is not None else False
        load_trainer = cfg.load_trainer_state if cfg.load_trainer_state is not None else False
        log.info(
            f"Loading checkpoint from '{cfg.load_path}' "
            f"(load_trainer_state={load_trainer}, load_optim_state={load_optim})..."
        )
        trainer.load_checkpoint(
            cfg.load_path,
            load_trainer_state=load_trainer,
            load_optim_state=load_optim,
        )

    trainer.fit()

    teardown_training_environment()


if __name__ == "__main__":
    main()
