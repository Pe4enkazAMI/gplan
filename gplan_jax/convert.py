"""Weight conversion between PyTorch and Equinox.

    lewm_from_torch(wm)          trained PyTorch LeWM (jepa.JEPA)  -> LeWMPredictor (JAX)
    gplaner_from_torch(model)    PyTorch gplan.policy.GPlaner      -> gplan_jax.policy.GPlaner
    gplaner_to_torch(model)      gplan_jax.policy.GPlaner          -> PyTorch state_dict (same keys as gplan.policy.GPlaner)

PyTorch and Equinox store a Linear weight the same way, (out_features, in_features), so arrays are
copied as they are. This module imports torch; the JAX training code only uses it at startup and
when saving checkpoints, on CPU.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from gplan_jax import lewm as jl
from gplan_jax.policy import GPlaner

_KEY = jax.random.key(0)  # layers are created with throwaway random weights, then overwritten


def _arr(t):
    return jnp.asarray(t.detach().cpu().float().numpy())


def linear(layer: torch.nn.Linear) -> eqx.nn.Linear:
    out = eqx.nn.Linear(layer.in_features, layer.out_features, use_bias=layer.bias is not None, key=_KEY)
    out = eqx.tree_at(lambda l: l.weight, out, _arr(layer.weight))
    if layer.bias is not None:
        out = eqx.tree_at(lambda l: l.bias, out, _arr(layer.bias))
    return out


def linear_or_none(layer):
    """le-wm uses nn.Identity where a projection is not needed; Identity becomes None."""
    return linear(layer) if isinstance(layer, torch.nn.Linear) else None


def layer_norm(norm: torch.nn.LayerNorm) -> eqx.nn.LayerNorm:
    affine = norm.weight is not None
    out = eqx.nn.LayerNorm(norm.normalized_shape, eps=norm.eps, use_weight=affine, use_bias=affine)
    if affine:
        out = eqx.tree_at(lambda l: (l.weight, l.bias), out, (_arr(norm.weight), _arr(norm.bias)))
    return out


# ----------------------------------------------------------------------------- LeWM

def _block(b) -> jl.ConditionalBlock:
    attn = b.attn
    to_out = attn.to_out[0] if isinstance(attn.to_out, torch.nn.Sequential) else attn.to_out
    return jl.ConditionalBlock(
        attn=jl.Attention(norm=layer_norm(attn.norm), to_qkv=linear(attn.to_qkv), to_out=linear_or_none(to_out),
                          heads=attn.heads),
        mlp=jl.FeedForward(norm=layer_norm(b.mlp.net[0]), fc1=linear(b.mlp.net[1]), fc2=linear(b.mlp.net[4])),
        norm1=layer_norm(b.norm1),
        norm2=layer_norm(b.norm2),
        ada_ln=linear(b.adaLN_modulation[1]),
    )


def lewm_from_torch(wm, history_size=None) -> jl.LeWMPredictor:
    """Convert the planning part of a trained PyTorch LeWM (`jepa.JEPA`, eval mode) to Equinox.

    `history_size` defaults to the predictor's context length (its number of position embeddings, 3 for LeWM)."""
    enc, pred, proj = wm.action_encoder, wm.predictor, wm.pred_proj
    conv = enc.patch_embed  # Conv1d(kernel_size=1) == Linear on each step
    patch = eqx.nn.Linear(conv.in_channels, conv.out_channels, key=_KEY)
    patch = eqx.tree_at(lambda l: (l.weight, l.bias), patch, (_arr(conv.weight[:, :, 0]), _arr(conv.bias)))

    tr = pred.transformer
    bn = proj.net[1]
    return jl.LeWMPredictor(
        action_encoder=jl.ActionEncoder(patch_embed=patch, fc1=linear(enc.embed[0]), fc2=linear(enc.embed[2])),
        predictor=jl.ARPredictor(
            pos_embedding=_arr(pred.pos_embedding[0]),
            input_proj=linear_or_none(tr.input_proj),
            cond_proj=linear_or_none(tr.cond_proj),
            blocks=[_block(b) for b in tr.layers],
            norm=layer_norm(tr.norm),
            output_proj=linear_or_none(tr.output_proj),
        ),
        pred_proj=jl.ProjectorMLP(
            fc1=linear(proj.net[0]), bn_mean=_arr(bn.running_mean), bn_var=_arr(bn.running_var),
            bn_weight=_arr(bn.weight), bn_bias=_arr(bn.bias), fc2=linear(proj.net[3]), bn_eps=bn.eps,
        ),
        history_size=history_size or pred.pos_embedding.shape[1],
    )


# ----------------------------------------------------------------------------- GPlaner

def _linear_params(prefix, layer):
    return {f"{prefix}.weight": layer.weight, f"{prefix}.bias": layer.bias}


def gplaner_to_torch(model: GPlaner) -> dict:
    """JAX GPlaner -> PyTorch state_dict with exactly the keys of `gplan.policy.GPlaner`."""
    params = {**_linear_params("in_proj.0", model.in_proj),
              **_linear_params("out_proj", model.out_proj),
              **_linear_params("Z.0", model.Z[0]),
              **_linear_params("Z.2", model.Z[1])}
    for i, layer in enumerate(model.backbone):
        params.update(_linear_params(f"backbone.{i}", layer))
    return {k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in params.items()}


def gplaner_from_torch(torch_model, key=_KEY) -> GPlaner:
    """PyTorch `gplan.policy.GPlaner` -> JAX GPlaner with identical weights and hyperparameters."""
    width, in_dim = torch_model.in_proj[0].weight.shape
    state_dim = torch_model.Z[0].in_features // 2
    model = GPlaner(state_dim=state_dim, horizon=torch_model.horizon, hidden_size=width // 3,
                    n_layers=len(torch_model.backbone), log_std_min=torch_model.log_std_min,
                    log_std_max=torch_model.log_std_max, action_dim=torch_model.action_dim, key=key)
    return eqx.tree_at(
        lambda m: (m.in_proj, m.backbone, m.out_proj, m.Z),
        model,
        (linear(torch_model.in_proj[0]), [linear(l) for l in torch_model.backbone], linear(torch_model.out_proj),
         [linear(torch_model.Z[0]), linear(torch_model.Z[2])]),
    )
