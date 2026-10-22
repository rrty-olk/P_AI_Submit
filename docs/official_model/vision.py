"""Images to prefix embeddings: LFM2-VL's tiling, a SigLIP2 NaFlex tower, and a 2x2 pixel-unshuffle projector.

A large image is cut into up to ten 512 px tiles plus a thumbnail; a small one is read whole. Each crop becomes
(h/32) x (w/32) prefix embeddings, at most 256 per crop. Several images are concatenated in the order given.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TILE, PATCH, MAX_PATCHES = 512, 16, 1024


def layout(width: int, height: int) -> dict:
    """LFM2-VL's smart resize, tile grid and thumbnail (projected-patch budget 64 to 256 per crop)."""
    if min(width, height) < 1:
        raise ValueError("empty image")
    factor, maximum, minimum = 32, 256 * 1024, 64 * 1024
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    large = max(16, round(height / factor) * factor) * max(16, round(width / factor) * factor) > maximum * 2
    grid = (1, 1)
    if large:
        ratios = sorted({(x, y) for n in range(2, 11) for x in range(1, n + 1) for y in range(1, n + 1)
                         if 2 <= x * y <= 10}, key=lambda r: r[0] * r[1])
        best = float("inf")
        for ratio in ratios:
            diff = abs(width / height - ratio[0] / ratio[1])
            if diff < best or (diff == best and width * height > 0.5 * TILE * TILE * ratio[0] * ratio[1]):
                grid, best = ratio, diff
    return {"grid": grid, "thumbnail": (h, w), "tiled": large}


def preprocess(image) -> dict:
    """A PIL image -> SigLIP2 NaFlex inputs for its tiles and thumbnail (RGB in [-1, 1], 16 px patches)."""
    from torchvision.transforms.v2 import functional as tvf

    image = image.convert("RGB")
    plan = layout(*image.size)
    x = tvf.pil_to_tensor(image)
    crops = []
    if plan["tiled"]:
        gw, gh = plan["grid"]
        big = tvf.resize(x, [gh * TILE, gw * TILE], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
        crops = [big[:, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for r in range(gh) for c in range(gw)]
    crops.append(tvf.resize(x, list(plan["thumbnail"]), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True))
    pixels, shapes, masks = [], [], []
    for crop in crops:
        crop = tvf.normalize(crop.to(torch.float32), [127.5] * 3, [127.5] * 3)
        _, h, w = crop.shape
        ph, pw = h // PATCH, w // PATCH
        patches = crop.reshape(3, ph, PATCH, pw, PATCH).permute(1, 3, 2, 4, 0).reshape(ph * pw, 3 * PATCH * PATCH)
        pixels.append(F.pad(patches, (0, 0, 0, MAX_PATCHES - ph * pw)))
        shapes.append([ph, pw])
        masks.append(torch.arange(MAX_PATCHES) < ph * pw)
    return {"pixel_values": torch.stack(pixels), "spatial_shapes": torch.tensor(shapes),
            "pixel_attention_mask": torch.stack(masks).to(torch.int32)}


def prefix_length(inputs: dict) -> int:
    return sum(h * w // 4 for h, w in inputs["spatial_shapes"].tolist())


class Projector(nn.Module):
    """LFM2-VL's 2x2 pixel unshuffle and two-layer GELU MLP into the trunk's width."""

    def __init__(self, vision_dim: int, hidden: int, out: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.linear_1 = nn.Linear(vision_dim * factor**2, hidden)
        self.linear_2 = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        f = self.factor
        x = x.reshape(b, h, w // f, c * f).permute(0, 2, 1, 3)
        x = x.reshape(b, w // f, h // f, c * f * f).permute(0, 2, 1, 3)
        return self.linear_2(F.gelu(self.linear_1(x))).reshape(b, -1, self.linear_2.out_features)


class Vision(nn.Module):
    def __init__(self, vision_config: dict, projector_hidden: int, out: int):
        super().__init__()
        from transformers import Siglip2VisionConfig, Siglip2VisionModel

        cfg = Siglip2VisionConfig(**vision_config)
        cfg._attn_implementation = "sdpa"
        self.tower = Siglip2VisionModel(cfg)
        self.projector = Projector(cfg.hidden_size, projector_hidden, out)

    def forward(self, images: list) -> torch.Tensor:
        """PIL images -> (1, P, D) prefix embeddings, every image's tiles then thumbnail, images in order."""
        param = next(self.tower.parameters())
        out = []
        for image in images:
            inputs = preprocess(image)
            inputs = {k: v.to(param.device, dtype=param.dtype if k == "pixel_values" else v.dtype)
                      for k, v in inputs.items()}
            hidden = self.tower(**inputs).last_hidden_state
            for i, (h, w) in enumerate(inputs["spatial_shapes"].tolist()):
                out.append(self.projector(hidden[i:i + 1, :h * w].reshape(1, h, w, -1)))
        return torch.cat(out, dim=1)
