"""Model factory. Both models map (B,3,H,W) -> (B,C,H,W) logits at input resolution."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_research.config import Config


class SegFormerWrapper(nn.Module):
    """HF SegFormer emits logits at 1/4 resolution; upsample bilinearly so the loss and metrics see full size."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(cfg: Config, num_classes: int) -> nn.Module:
    if cfg.model == "unet_r50":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name=cfg.encoder,
            encoder_weights="imagenet" if cfg.pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
    if cfg.model == "segformer_b2":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if cfg.smoke:  # tiny random SegFormer: no download
            hf_cfg = SegformerConfig(
                num_labels=num_classes, depths=[1, 1, 1, 1], hidden_sizes=[16, 32, 64, 128],
                num_attention_heads=[1, 2, 4, 8], decoder_hidden_size=32,
            )
            return SegFormerWrapper(SegformerForSemanticSegmentation(hf_cfg))
        if cfg.pretrained:
            # mit-b2 checkpoint = ImageNet-1k pretrained encoder only; the all-MLP decode head is freshly initialised.
            net = SegformerForSemanticSegmentation.from_pretrained(cfg.segformer_checkpoint, num_labels=num_classes)
        else:
            hf_cfg = SegformerConfig.from_pretrained(cfg.segformer_checkpoint, num_labels=num_classes)
            net = SegformerForSemanticSegmentation(hf_cfg)
        return SegFormerWrapper(net)
    raise ValueError(f"unknown model {cfg.model!r}")


def count_params(model: nn.Module) -> dict:
    return {
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
