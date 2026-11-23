"""Convert world-model weights of the LeWM architecture (a state_dict + its config.json) to the
`<name>_object.ckpt` (a pickled `jepa.JEPA`) that `stable_worldmodel` and our train / eval scripts load:
LeWM weights from the Hugging Face Hub, or a world model trained with stable-worldmodel's scripts
(e.g. PLDM, scripts/train/pldm.py), from its checkpoint folder.

    python scripts/convert_lewm.py --repo quentinll/lewm-tworooms --name tworoom/lewm
    -> $STABLEWM_HOME/tworoom/lewm_object.ckpt   (usable as swm.policy.AutoCostModel("tworoom/lewm"))

    python scripts/convert_lewm.py --run-dir $STABLEWM_HOME/checkpoints/tworoom/pldm --name tworoom/pldm
    -> $STABLEWM_HOME/tworoom/pldm_object.ckpt   (the run's last weights_epoch_<n>.pt; --weights picks another)

Follows the recipe in le-wm/README.md, except that the ViT is built directly with
`transformers` (identical to `stable_pretraining.backbone.utils.vit_hf`) so the
script does not depend on `stable_pretraining` being importable.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel

from gplan.torch.lewm import STABLEWM_HOME  # also puts the le-wm submodule on sys.path

# isort: split
from jepa import JEPA  # le-wm modules, importable once gplan.torch.lewm is imported
from module import ARPredictor, Embedder, MLP

VIT_SIZES = {  # as in stable_pretraining.backbone.utils.vit_hf
    "tiny": dict(hidden_size=192, num_hidden_layers=12, num_attention_heads=3),
    "small": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6),
    "base": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12),
    "large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16),
}


def build_encoder(cfg):
    assert not cfg.get("pretrained", False), "LeWM encoders are trained from scratch"
    size = VIT_SIZES[cfg["size"]]
    vit_cfg = ViTConfig(**size, intermediate_size=4 * size["hidden_size"],
                        image_size=cfg["image_size"], patch_size=cfg["patch_size"])
    encoder = ViTModel(vit_cfg, add_pooling_layer=False, use_mask_token=cfg.get("use_mask_token", False))
    encoder.config.interpolate_pos_encoding = True
    return encoder


def build_lewm(config):
    """Instantiate a le-wm `JEPA` from the HF `config.json` (hydra-style dict with `_target_` keys)."""
    cfg = {k: {kk: vv for kk, vv in v.items() if kk != "_target_"} for k, v in config.items() if isinstance(v, dict)}
    mlp = lambda c: MLP(input_dim=c["input_dim"], output_dim=c["output_dim"], hidden_dim=c["hidden_dim"],
                        norm_fn=torch.nn.BatchNorm1d)
    return JEPA(
        encoder=build_encoder(cfg["encoder"]),
        predictor=ARPredictor(**cfg["predictor"]),
        action_encoder=Embedder(**cfg["action_encoder"]),
        projector=mlp(cfg["projector"]),
        pred_proj=mlp(cfg["pred_proj"]),
    )


# transformers >= 5 names the ViT's weights differently (stable-worldmodel's training environment has it);
# the ViT built here (transformers 4) computes the same with its own names
VIT_V5_NAMES = [
    (r"^encoder\.layers\.(\d+)\.attention\.([qkv])_proj\.",
     lambda m: f"encoder.encoder.layer.{m[1]}.attention.attention.{dict(q='query', k='key', v='value')[m[2]]}."),
    (r"^encoder\.layers\.(\d+)\.attention\.o_proj\.", r"encoder.encoder.layer.\1.attention.output.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc1\.", r"encoder.encoder.layer.\1.intermediate.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc2\.", r"encoder.encoder.layer.\1.output.dense."),
    (r"^encoder\.layers\.(\d+)\.(layernorm_before|layernorm_after)\.", r"encoder.encoder.layer.\1.\2."),
]


def vit_v4_names(state_dict):
    """A state_dict whose ViT weights may carry transformers-5 names, with the transformers-4 ones."""
    out = {}
    for key, value in state_dict.items():
        for pattern, repl in VIT_V5_NAMES:
            key = re.sub(pattern, repl, key)
        out[key] = value
    return out


def last_epoch(run_dir):
    """The weights_epoch_<n>.pt with the largest n in a stable-worldmodel checkpoint folder."""
    weights = sorted(Path(run_dir).glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not weights:
        raise SystemExit(f"no weights_epoch_<n>.pt in {run_dir}")
    return weights[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="quentinll/lewm-tworooms", help="HF model repo with weights.pt + config.json")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="instead of --repo: a stable-worldmodel checkpoint folder (config.json, weights_epoch_<n>.pt)")
    p.add_argument("--weights", type=Path, default=None, help="with --run-dir: this weights file (default: the last epoch)")
    p.add_argument("--name", default="tworoom/lewm", help="checkpoint name relative to $STABLEWM_HOME")
    args = p.parse_args()

    out = STABLEWM_HOME / f"{args.name}_object.ckpt"
    if args.run_dir is not None:
        config, weights = args.run_dir / "config.json", args.weights or last_epoch(args.run_dir)
    else:
        from huggingface_hub import hf_hub_download
        src = STABLEWM_HOME / f"hf_{args.name.replace('/', '_')}"
        weights = hf_hub_download(args.repo, "weights.pt", local_dir=src)
        config = hf_hub_download(args.repo, "config.json", local_dir=src)
    print(f"converting {weights}")

    model = build_lewm(json.loads(Path(config).read_text()))
    state_dict = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(vit_v4_names(state_dict), strict=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.eval(), out)
    print(f"saved {out} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")


if __name__ == "__main__":
    main()
