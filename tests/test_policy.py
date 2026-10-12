import pytest
import torch
from conftest import A, B, D, T

from gplan.policy import GPlaner, Sampler, load_planner, save_planner


@pytest.mark.parametrize("action_dim, n_components", [(1, 1), (A, 1), (A, 3)])
def test_rollout_and_log_prob_shapes(action_dim, n_components):
    """Scalar (action_dim=1) and blocked layouts, Gaussian or mixture: a full buffer and per-step log-probs."""
    torch.manual_seed(0)
    s = Sampler(GPlaner(state_dim=D, horizon=T * A, action_dim=action_dim, n_components=n_components))
    z_start, z_goal = torch.randn(B, D), torch.randn(B, D)
    actions = s.rollout(z_start, z_goal)
    assert actions.shape == (B, T * A)
    assert s.log_prob(z_start, z_goal, actions).shape == (B, T * A // action_dim)


def test_log_prob_ignores_future_slots():
    """Teacher forcing: the score of step t must not depend on steps >= t."""
    torch.manual_seed(0)
    s = Sampler(GPlaner(state_dim=D, horizon=T * A, action_dim=A))
    z_start, z_goal = torch.randn(B, D), torch.randn(B, D)
    actions = s.rollout(z_start, z_goal)
    perturbed = actions.clone()
    perturbed[:, A:] += 1.0  # change every step after the first
    assert torch.allclose(s.log_prob(z_start, z_goal, actions)[:, 0], s.log_prob(z_start, z_goal, perturbed)[:, 0])


def test_log_z_keeps_batch_dim_for_single_condition():
    model = GPlaner(state_dim=D, horizon=T * A, action_dim=A)
    assert model.log_Z(torch.randn(1, D), torch.randn(1, D)).shape == (1,)


@pytest.mark.parametrize("bare, n_components", [(False, 1), (False, 3), (True, 1)])
def test_checkpoint_roundtrip(tmp_path, bare, n_components):
    kwargs = dict(state_dim=D, horizon=T * A, hidden_size=8, n_layers=3, action_dim=A, n_components=n_components)
    model = GPlaner(**kwargs)
    path = tmp_path / "planner.pt"
    if bare:
        torch.save(model.state_dict(), path)
    else:
        save_planner(path, model, kwargs, train_args={"n_steps": T})
    loaded, train_args = load_planner(path)
    assert train_args == (None if bare else {"n_steps": T})
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k
