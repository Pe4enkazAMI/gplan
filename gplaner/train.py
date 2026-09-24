"""Train the GPlaner sampler with the Trajectory Balance loss.

    loss = ( log Z(c) + sum_t log P_F(a_t | a_<t, c) + beta * J(A, c) )^2

where c = (z_start, z_goal) are embeddings from a frozen LeWorldModel and
J(A, c) is the LeWM planning cost: squared distance between the embedding the
world model predicts after executing the actions A and the goal embedding.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the compression filter used by the LeWM .h5 files
import numpy as np
import torch
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                    # gflow
sys.path.insert(0, str(ROOT.parent / "le-wm"))   # jepa / module, needed to unpickle the LeWM ckpt

from gflow import GPlaner, Sampler  # noqa: E402

STABLEWM_HOME = Path(os.getenv("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser()
IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def img_transform(img_size=224):
    """uint8 HWC frame -> normalized float CHW tensor, as LeWM was trained."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**IMAGENET),
        transforms.Resize(size=img_size),
    ])


def valid_start_rows(ep_idx, step_idx, goal_offset):
    """Dataset rows with at least `goal_offset` steps left in their episode (so row + goal_offset is the goal)."""
    ep_len = np.zeros(ep_idx.max() + 1, dtype=np.int64)
    np.maximum.at(ep_len, ep_idx, step_idx + 1)
    return np.nonzero(step_idx <= ep_len[ep_idx] - goal_offset - 1)[0]


def make_batches(n_batches, batch_size, img_size=224, dataset="tworoom", goal_offset=25, seed=0):
    """Yield (start_pixels, goal_pixels) batches of shape (B, 3, img_size, img_size) from a LeWM .h5 dataset.

    The start is a random dataset frame and the goal is the frame `goal_offset`
    steps later in the same episode, matching the evaluation protocol.
    """
    h5 = h5py.File(STABLEWM_HOME / f"{dataset}.h5", "r")
    ep_col = "episode_idx" if "episode_idx" in h5 else "ep_idx"
    valid = valid_start_rows(h5[ep_col][:], h5["step_idx"][:], goal_offset)
    transform = img_transform(img_size)
    rng = np.random.default_rng(seed)

    for _ in range(n_batches):
        rows = np.sort(rng.choice(valid, size=batch_size, replace=False))  # h5 needs sorted indices
        start = torch.stack([transform(f) for f in h5["pixels"][rows]])
        goal = torch.stack([transform(f) for f in h5["pixels"][rows + goal_offset]])
        yield start, goal


def load_lewm(ckpt_path, device="cpu"):
    """Load a LeWM `*_object.ckpt` (a pickled `jepa.JEPA`) as a frozen eval model."""
    wm = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm.eval().requires_grad_(False)
    return wm


@torch.no_grad()
def encode(wm, pixels):
    """Frozen LeWM embedding of a batch of frames. pixels: (B, C, H, W) -> (B, D)."""
    return wm.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]


@torch.no_grad()
def lewm_cost(wm, z_start, z_goal, actions, history_size=3):
    """J(A, c): LeWM planning cost of an action plan.

    Rolls the frozen predictor forward from z_start through `actions` and returns
    the squared distance of the final predicted embedding to z_goal. This is
    `JEPA.rollout` + `JEPA.criterion` with a single start frame, run directly
    on embeddings.

    Args:
        z_start, z_goal: (B, D)
        actions:         (B, T, A) actions in the world model's action space.
    Returns:
        (B,) cost per plan.
    """
    emb = z_start.unsqueeze(1)  # (B, 1, D)
    for t in range(actions.shape[1]):
        act_emb = wm.action_encoder(actions[:, : t + 1])
        pred = wm.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
    return (emb[:, -1] - z_goal).pow(2).sum(-1)


def tb_loss(sampler, wm, z_start, z_goal, actions, beta, action_dim):
    """Trajectory Balance loss for a batch of sampled action buffers.

    Args:
        actions:    (B, T * action_dim) buffer from `sampler.rollout`.
        beta:       temperature multiplying the cost (log R = -beta * J).
        action_dim: dimension of a single world-model action.
    """
    B = actions.shape[0]
    log_pf = sampler.log_prob(z_start, z_goal, actions).sum(-1)   # (B,)
    log_z = sampler.model.log_Z(z_start, z_goal)                  # (B,)
    cost = lewm_cost(wm, z_start, z_goal, actions.view(B, -1, action_dim))  # (B,)
    return (log_z + log_pf + beta * cost).pow(2).mean()


def train(sampler, wm, batches, beta, action_dim, lr=1e-3, device="cpu", log_every=10):
    """Run one optimization step per (start_pixels, goal_pixels) batch."""
    opt = torch.optim.Adam(sampler.model.parameters(), lr=lr)
    losses = []
    for step, (start_pixels, goal_pixels) in enumerate(batches):
        z_start = encode(wm, start_pixels.to(device))
        z_goal = encode(wm, goal_pixels.to(device))

        actions = sampler.rollout(z_start, z_goal)
        loss = tb_loss(sampler, wm, z_start, z_goal, actions, beta, action_dim)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step % log_every == 0:
            print(f"step {step:5d}  tb_loss {loss.item():.4f}")
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=STABLEWM_HOME / "tworoom/lewm_object.ckpt", help="LeWM *_object.ckpt")
    p.add_argument("--dataset", default="tworoom", help=".h5 name under $STABLEWM_HOME")
    p.add_argument("--goal-offset", type=int, default=25, help="goal = start + this many dataset steps")
    p.add_argument("--n-steps", type=int, default=5, help="planning horizon T")
    p.add_argument("--action-dim", type=int, default=10, help="frameskip * env action dim")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=1000)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=ROOT / "gplaner.pt", help="where to save the trained GPlaner state_dict")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    wm = load_lewm(args.ckpt, args.device)
    embed_dim = wm.predictor.pos_embedding.shape[-1]
    model = GPlaner(state_dim=embed_dim, horizon=args.n_steps * args.action_dim).to(args.device)
    sampler = Sampler(model)

    batches = make_batches(args.n_batches, args.batch_size, args.img_size, args.dataset, args.goal_offset, args.seed)
    train(sampler, wm, batches, args.beta, args.action_dim, args.lr, args.device)
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
