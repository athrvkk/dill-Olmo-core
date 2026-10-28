import torch
import torch.nn as nn

from olmo_core.optim import KronConfig, OptimGroupOverride
from olmo_core.optim.kron import Kron, _ProbScheduler, _merge_dims


class KronTestModel(nn.Module):
    """Small model for testing Kron optimizer."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(128, 16)
        self.fc1 = nn.Linear(16, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.ln = nn.LayerNorm(16)
        self.head = nn.Linear(16, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        return self.head(x)


def test_config_builds_correctly():
    """Test that KronConfig.build() creates the right optimizer type."""
    config = KronConfig(lr=1e-3)
    model = KronTestModel()
    optim = config.build(model)

    assert isinstance(optim, Kron)
    assert len(optim.param_groups) == 1

    for group in optim.param_groups:
        assert "initial_lr" in group


def test_config_with_group_overrides():
    """Test parameter group overrides work."""
    config = KronConfig(
        group_overrides=[OptimGroupOverride(params=["wte.*"], opts=dict(weight_decay=0.0))]
    )
    model = KronTestModel()
    optim = config.build(model)

    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.0


def test_optimizer_step():
    """Test that the optimizer updates parameters."""
    config = KronConfig()
    model = KronTestModel().train()
    optim = config.build(model)

    initial_params = {n: p.clone() for n, p in model.named_parameters()}
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 128, (2, 8))).sum().backward()
    optim.step()

    any_changed = False
    for n, p in model.named_parameters():
        if not torch.equal(p, initial_params[n]):
            any_changed = True
            break
    assert any_changed, "No parameters were updated after optim.step()"


def test_preconditioner_init():
    """Test that Q factors are initialized correctly."""
    config = KronConfig()
    model = KronTestModel().train()
    optim = config.build(model)

    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 128, (2, 8))).sum().backward()
    optim.step()

    for p in model.parameters():
        if p in optim.state:
            state = optim.state[p]
            assert "Q" in state, "Q factors not initialized"
            assert "exprs" in state, "Einsum expressions not initialized"
            assert "momentum_buffer" in state, "Momentum buffer not initialized"
            for q in state["Q"]:
                assert q.norm().item() > 0, "Q factor is all zeros"


def test_precond_update_schedule():
    """Test that the probabilistic schedule anneals."""
    sched = _ProbScheduler(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500)

    # Before flat_start, should be max_prob.
    assert sched(0) == 1.0
    assert sched(499) == 1.0

    # After flat_start, should decay.
    p_at_1000 = sched(1000)
    p_at_2000 = sched(2000)
    assert p_at_1000 < 1.0
    assert p_at_2000 < p_at_1000
    assert p_at_2000 >= 0.03


def test_multiple_steps():
    """Test that multiple consecutive steps work without error."""
    config = KronConfig()
    model = KronTestModel().train()
    optim = config.build(model)

    for _ in range(10):
        optim.zero_grad(set_to_none=True)
        model(torch.randint(0, 128, (2, 8))).sum().backward()
        optim.step()


def test_merge_dims():
    """Test dimension merging for >2D tensors."""
    # 3D shape should be merged.
    result = _merge_dims((4, 8, 16))
    assert result is not None
    assert len(result) <= 2

    # 2D shape should return None (no merging needed).
    assert _merge_dims((32, 64)) is None

    # 1D shape should return None.
    assert _merge_dims((128,)) is None


def test_state_dict_roundtrip():
    """Test that state_dict / load_state_dict preserves optimizer state."""
    config = KronConfig()
    model = KronTestModel().train()
    optim = config.build(model)

    # Take steps to populate state and advance the scheduler.
    for _ in range(5):
        optim.zero_grad(set_to_none=True)
        model(torch.randint(0, 128, (2, 8))).sum().backward()
        optim.step()

    assert optim._prob_step == 5

    sd = optim.state_dict()
    assert all(s["_kron_prob_step"] == 5 for s in sd["state"].values())

    # Reset and reload.
    original_prob_step = optim._prob_step
    optim._prob_step = 0
    optim._update_counter = 0.0
    optim.load_state_dict(sd)

    assert optim._prob_step == original_prob_step

    # Verify initial_lr is preserved.
    for g in optim.param_groups:
        assert g["initial_lr"] == config.lr


def test_flattened_state_dict_roundtrip():
    """Test that state survives the flattened optimizer state dict used for checkpoints."""
    import copy

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    config = KronConfig()
    model = KronTestModel().train()
    optim = config.build(model)

    def train_step(m, o, seed):
        torch.manual_seed(seed)
        o.zero_grad(set_to_none=True)
        m(torch.randint(0, 128, (2, 8))).sum().backward()
        o.step()

    for i in range(3):
        train_step(model, optim, i)

    opts = StateDictOptions(flatten_optimizer_state_dict=True)
    sd = get_optimizer_state_dict(model, optim, options=opts)
    assert any(".Q." in k for k in sd), "expected Q factors in the flattened state dict"

    # Restore into a fresh copy and check both continue identically.
    model2 = copy.deepcopy(model)
    optim2 = config.build(model2)
    train_step(model2, optim2, 0)  # populate state so it has the right structure
    model2.load_state_dict(model.state_dict())
    # Deep copy: the state dict holds references to optim's live tensors.
    set_optimizer_state_dict(model2, optim2, copy.deepcopy(sd), options=opts)

    assert optim2._prob_step == optim._prob_step
    for p in model2.parameters():
        assert isinstance(optim2.state[p]["Q"], list)
        assert "exprs" in optim2.state[p]

    train_step(model, optim, 10)
    train_step(model2, optim2, 10)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)
