"""Download LeWM weights from the Hugging Face Hub and convert them to the
`<name>_object.ckpt` (a pickled `jepa.JEPA`) that `stable_worldmodel` and our
train / eval scripts load.

    python gplaner/convert_lewm.py --repo quentinll/lewm-tworooms --name tworoom/lewm
    -> $STABLEWM_HOME/tworoom/lewm_object.ckpt   (usable as swm.policy.AutoCostModel("tworoom/lewm"))

Follows the recipe in le-wm/README.md, except that the ViT is built directly with
`transformers` (identical to `stable_pretraining.backbone.utils.vit_hf`) so the
script does not depend on `stable_pretraining` being importable.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import ViTConfig, ViTModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import STABLEWM_HOME  # noqa: E402  (puts le-wm on sys.path)
from jepa import JEPA  # noqa: E402
from module import MLP, ARPredictor, Embedder  # noqa: E402

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="quentinll/lewm-tworooms", help="HF model repo with weights.pt + config.json")
    p.add_argument("--name", default="tworoom/lewm", help="checkpoint name relative to $STABLEWM_HOME")
    args = p.parse_args()

    src = STABLEWM_HOME / f"hf_{args.name.replace('/', '_')}"
    out = STABLEWM_HOME / f"{args.name}_object.ckpt"

    weights = hf_hub_download(args.repo, "weights.pt", local_dir=src)
    config = hf_hub_download(args.repo, "config.json", local_dir=src)

    model = build_lewm(json.loads(Path(config).read_text()))
    state_dict = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.eval(), out)
    print(f"saved {out} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")


if __name__ == "__main__":
    main()
