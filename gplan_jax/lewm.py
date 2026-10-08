"""LeWorldModel predictor in Equinox: action encoder + autoregressive predictor + output MLP.

Inference only. Weights always come from a trained PyTorch LeWM (see `convert.lewm_from_torch`);
this module mirrors `le-wm/module.py` and `le-wm/jepa.py` layer by layer. Every module works on
one sequence; batch with `jax.vmap`.

Details that must match PyTorch exactly:
  * GELU is the exact (erf) version, not JAX's default tanh approximation;
  * attention and feed-forward LayerNorms have affine params and eps 1e-5, the AdaLN norms
    have no params and eps 1e-6;
  * the BatchNorm in the output MLP runs in eval mode (running statistics).
"""

import equinox as eqx
import jax
import jax.numpy as jnp


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def apply_seq(layer, x):
    """Apply a per-vector layer (Linear, LayerNorm) to every row of x: (T, d_in) -> (T, d_out)."""
    return jax.vmap(layer)(x)


class Attention(eqx.Module):
    """Causal multi-head self-attention with a pre-LayerNorm (le-wm `Attention`)."""
    norm: eqx.nn.LayerNorm
    to_qkv: eqx.nn.Linear
    to_out: eqx.nn.Linear | None
    heads: int = eqx.field(static=True)

    def __call__(self, x):  # (T, D) -> (T, D)
        T = x.shape[0]
        qkv = apply_seq(self.to_qkv, apply_seq(self.norm, x))
        q, k, v = (t.reshape(T, self.heads, -1).transpose(1, 0, 2) for t in jnp.split(qkv, 3, axis=-1))  # (H, T, d)
        scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(q.shape[-1])
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        out = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1) @ v
        out = out.transpose(1, 0, 2).reshape(T, -1)
        return out if self.to_out is None else apply_seq(self.to_out, out)


class FeedForward(eqx.Module):
    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __call__(self, x):  # (T, D) -> (T, D)
        return apply_seq(self.fc2, gelu(apply_seq(self.fc1, apply_seq(self.norm, x))))


class ConditionalBlock(eqx.Module):
    """Transformer block with AdaLN-zero conditioning on the action embedding."""
    attn: Attention
    mlp: FeedForward
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    ada_ln: eqx.nn.Linear  # SiLU -> Linear(D, 6D)

    def __call__(self, x, c):  # x, c: (T, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            apply_seq(self.ada_ln, jax.nn.silu(c)), 6, axis=-1)
        h = apply_seq(self.norm1, x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(h)
        h = apply_seq(self.norm2, x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(h)


class ARPredictor(eqx.Module):
    """Next-embedding predictor: position embedding + stack of conditional blocks + final LayerNorm."""
    pos_embedding: jax.Array  # (num_frames, D)
    input_proj: eqx.nn.Linear | None
    cond_proj: eqx.nn.Linear | None
    blocks: list[ConditionalBlock]
    norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear | None

    def __call__(self, x, c):  # x: (T, D) embeddings, c: (T, D) action embeddings
        x = x + self.pos_embedding[: x.shape[0]]
        if self.input_proj is not None:
            x = apply_seq(self.input_proj, x)
        if self.cond_proj is not None:
            c = apply_seq(self.cond_proj, c)
        for block in self.blocks:
            x = block(x, c)
        x = apply_seq(self.norm, x)
        return x if self.output_proj is None else apply_seq(self.output_proj, x)


class ActionEncoder(eqx.Module):
    """le-wm `Embedder`: a per-step linear map (the kernel-1 Conv1d), then Linear-SiLU-Linear."""
    patch_embed: eqx.nn.Linear
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __call__(self, a):  # (T, A) -> (T, D)
        return apply_seq(self.fc2, jax.nn.silu(apply_seq(self.fc1, apply_seq(self.patch_embed, a))))


class ProjectorMLP(eqx.Module):
    """le-wm `MLP` with BatchNorm1d in eval mode: Linear -> BatchNorm -> GELU -> Linear."""
    fc1: eqx.nn.Linear
    bn_mean: jax.Array
    bn_var: jax.Array
    bn_weight: jax.Array
    bn_bias: jax.Array
    fc2: eqx.nn.Linear
    bn_eps: float = eqx.field(static=True)

    def __call__(self, x):  # (D,) -> (D,)
        h = self.fc1(x)
        h = (h - self.bn_mean) / jnp.sqrt(self.bn_var + self.bn_eps) * self.bn_weight + self.bn_bias
        return self.fc2(gelu(h))


class LeWMPredictor(eqx.Module):
    """The parts of LeWM that planning needs: everything except the image encoder."""
    action_encoder: ActionEncoder
    predictor: ARPredictor
    pred_proj: ProjectorMLP
    history_size: int = eqx.field(static=True)

    def predict(self, emb, act_emb):  # (T, D), (T, D) -> (T, D) predicted next embeddings
        return jax.vmap(self.pred_proj)(self.predictor(emb, act_emb))

    def plan_cost(self, z_start, z_goal, actions):
        """J(A, c) for one plan: roll the predictor from z_start through actions (T, A), return
        the squared distance of the final predicted embedding to z_goal. Same as `gplan.lewm.lewm_cost`."""
        act_emb = self.action_encoder(actions)  # per-step, so embedding the whole plan at once is equivalent
        embs = [z_start]
        for t in range(actions.shape[0]):
            lo_e = max(0, len(embs) - self.history_size)
            lo_a = max(0, t + 1 - self.history_size)
            pred = self.predict(jnp.stack(embs[lo_e:]), act_emb[lo_a:t + 1])[-1]
            embs.append(pred)
        return jnp.sum((embs[-1] - z_goal) ** 2)

    def cost(self, z_start, z_goal, plans):
        """Batched J: z_start, z_goal (N, D), plans (N, T, A) -> (N,)."""
        return jax.vmap(self.plan_cost)(z_start, z_goal, plans)
