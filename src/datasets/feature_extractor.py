"""On-demand multi-scale feature extraction and external caching for PRISM2 and MLLM-HWSI."""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from safetensors.torch import load_file
from scipy.ndimage import label as connected_components
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from src.utils.paths import ProjectPaths

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Coordinate discovery from WSI / SlideBench ROI annotations
# ---------------------------------------------------------------------------


def discover_tissue_patch_coords_256(wsi_path: Path, feature_csv_path: Optional[Path] = None) -> List[Tuple[int, int]]:
    """Return (x, y) top-left coordinates for 256x256 tissue patches of a slide.

    Uses exact ROI polygon bounding boxes and patch offsets when SlideBench JSON + CSV
    annotations exist (e.g. BCNB), or falls back to saturation-based foreground tiling.
    """
    json_path = wsi_path.with_suffix(".json")
    if json_path.is_file() and feature_csv_path is not None and feature_csv_path.is_file():
        with json_path.open("r", encoding="utf-8") as f:
            js = json.load(f)
        positives = js.get("positive", [])
        bboxes: List[Tuple[int, int]] = []
        for ann in positives:
            verts = np.asarray(ann["vertices"], dtype=np.int64)
            min_x, min_y = int(verts[:, 0].min()), int(verts[:, 1].min())
            bboxes.append((min_x, min_y))

        coords: List[Tuple[int, int]] = []
        with feature_csv_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            patch_name = line.rsplit(",", 1)[-1].strip()
            stem = Path(patch_name).stem
            parts = stem.split("_")
            if len(parts) >= 4:
                ann_id = int(parts[-3])
                off_x = int(parts[-2])
                off_y = int(parts[-1])
                if 0 <= ann_id < len(bboxes):
                    bx, by = bboxes[ann_id]
                    coords.append((bx + off_x, by + off_y))
        if coords:
            return coords

    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as f:
            js = json.load(f)
        coords = []
        for ann in js.get("positive", []):
            verts = np.asarray(ann["vertices"], dtype=np.int64)
            min_x, min_y = int(verts[:, 0].min()), int(verts[:, 1].min())
            max_x, max_y = int(verts[:, 0].max()), int(verts[:, 1].max())
            for x in range(min_x, max(min_x + 1, max_x - 255), 256):
                for y in range(min_y, max(min_y + 1, max_y - 255), 256):
                    coords.append((x, y))
        if coords:
            return coords

    with Image.open(wsi_path) as im:
        w, h = im.size
        scale = 16
        thumb = np.asarray(im.convert("RGB").resize((max(1, w // scale), max(1, h // scale))))
    sat = thumb.max(axis=-1).astype(np.int32) - thumb.min(axis=-1).astype(np.int32)
    coords = []
    step = 256 // scale
    for gy in range(0, thumb.shape[0] - step + 1, step):
        for gx in range(0, thumb.shape[1] - step + 1, step):
            if sat[gy : gy + step, gx : gx + step].mean() > 12.0:
                coords.append((gx * scale, gy * scale))
    if not coords:
        coords.append((max(0, w // 2 - 128), max(0, h // 2 - 128)))
    return coords


def discover_region_coords_4096(wsi_path: Path, region_size: int = 4096) -> List[Tuple[int, int]]:
    """Return (x, y) top-left coordinates for 4096x4096 tissue regions of a slide."""
    with Image.open(wsi_path) as im:
        w, h = im.size

    json_path = wsi_path.with_suffix(".json")
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as f:
            js = json.load(f)
        seen_grid = set()
        regions: List[Tuple[int, int]] = []
        for ann in js.get("positive", []):
            verts = np.asarray(ann["vertices"], dtype=np.int64)
            cx = int(verts[:, 0].mean())
            cy = int(verts[:, 1].mean())
            rx = max(0, min(max(0, w - region_size), (cx // 2048) * 2048 - region_size // 4))
            ry = max(0, min(max(0, h - region_size), (cy // 2048) * 2048 - region_size // 4))
            grid_key = (rx // 1024, ry // 1024)
            if grid_key not in seen_grid:
                seen_grid.add(grid_key)
                regions.append((rx, ry))
        if regions:
            return regions

    regions = []
    for y in range(0, max(1, h), region_size):
        for x in range(0, max(1, w), region_size):
            regions.append((min(x, max(0, w - region_size)), min(y, max(0, h - region_size))))
    return regions[:16]


def crop_with_padding(img: Image.Image, x: int, y: int, size: int) -> Image.Image:
    """Crop a square region of `size x size` from `img`, padding with white if out of bounds."""
    w, h = img.size
    if x >= 0 and y >= 0 and x + size <= w and y + size <= h:
        return img.crop((x, y, x + size, y + size)).convert("RGB")
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    rx0, ry0 = max(0, x), max(0, y)
    rx1, ry1 = min(w, x + size), min(h, y + size)
    if rx1 > rx0 and ry1 > ry0:
        sub = img.crop((rx0, ry0, rx1, ry1)).convert("RGB")
        canvas.paste(sub, (rx0 - x, ry0 - y))
    return canvas


# ---------------------------------------------------------------------------
# 1. PRISM2 Feature Extractor (Virchow2 1280-d CLS token)
# ---------------------------------------------------------------------------


class Virchow2FeatureExtractor:
    """Extracts per-slide (N, 1280) Virchow2 CLS token embeddings for PRISM2."""

    def __init__(self, paths: Optional[ProjectPaths] = None, device: str = "cuda:0") -> None:
        self.paths = paths or ProjectPaths()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        cfg_path = self.paths.virchow2_dir / "config.json"
        weights_path = self.paths.virchow2_dir / "model.safetensors"

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.model = timm.create_model(
            cfg["architecture"],
            pretrained=False,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            **cfg.get("model_args", {}),
        )
        state_dict = load_file(str(weights_path))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)
        self.transform = create_transform(
            **resolve_data_config(cfg.get("pretrained_cfg", {}), model=self.model)
        )

    @torch.inference_mode()
    def extract_or_load(
        self,
        wsi_path: Path,
        cohort: str,
        feature_csv_path: Optional[Path] = None,
        batch_size: int = 64,
    ) -> Path:
        """Return cached Virchow2 .pt path or extract and cache it in ../Datasets/."""
        out_dir = self.paths.datasets_dir / cohort / "Feat" / "PRISM2_Virchow2"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{wsi_path.stem}.pt"
        if out_path.is_file():
            return out_path

        coords = discover_tissue_patch_coords_256(wsi_path, feature_csv_path=feature_csv_path)
        with Image.open(wsi_path) as im:
            im_rgb = im.convert("RGB")
            tensors = [self.transform(crop_with_padding(im_rgb, x, y, 256)) for x, y in coords]

        cls_chunks: List[torch.Tensor] = []
        use_amp = self.device.type == "cuda"
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size], dim=0).to(self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                out = self.model(batch)  # (B, 261, 1280)
                cls_tok = out[:, 0]      # (B, 1280) class token only for PRISM2
            cls_chunks.append(cls_tok.detach().float().cpu())

        tile_embeddings = torch.cat(cls_chunks, dim=0)
        tmp_path = out_dir / f".{wsi_path.stem}.{os.getpid()}.pt"
        torch.save(tile_embeddings, tmp_path)
        tmp_path.replace(out_path)
        return out_path


# ---------------------------------------------------------------------------
# 2. MLLM-HWSI Encoders (HIPT-256, HIPT-4K, CONCH, CellViT-256, DINOv2-ViTS14)
# ---------------------------------------------------------------------------


class _HIPTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _HIPTMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _HIPTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _HIPTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _HIPTMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _HIPTPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class HIPTViT256(nn.Module):
    """DINO ViT-Small/16 for 256x256 patches with optional intermediate layer extraction."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__()
        self.patch_size = 16
        self.embed_dim = 384
        self.patch_embed = _HIPTPatchEmbed(16, 3, 384)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 384))
        self.blocks = nn.ModuleList([_HIPTBlock(384, 6, 4.0, True) for _ in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.head = nn.Linear(384, num_classes) if num_classes > 0 else nn.Identity()

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size + 0.1, h // self.patch_size + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x + self.interpolate_pos_encoding(x, w, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward_with_skip_layers(
        self, x: torch.Tensor, extract_layers: Tuple[int, ...] = (3, 6, 9, 12)
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        extracted = []
        x = self.prepare_tokens(x)
        for idx, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if idx in extract_layers:
                extracted.append(x)
        x = self.norm(x)
        return self.head(x[:, 0]), x[:, 0], extracted


class HIPTViT4K(nn.Module):
    """DINO ViT-4K region aggregator mapping (B, 384, 16, 16) -> (B, 192)."""

    def __init__(self) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(384, 192), nn.GELU(), nn.Dropout(0.0))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 192))
        self.blocks = nn.ModuleList([_HIPTBlock(192, 6, 4.0, True) for _ in range(6)])
        self.norm = nn.LayerNorm(192, eps=1e-6)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, w, h = x.shape
        x = x.flatten(2, 3).transpose(1, 2)
        x = self.phi(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class ConchVisualEncoder(nn.Module):
    """Pure-PyTorch + timm wrapper for CONCH ViT-B/16 visual contrastive branch (512-d)."""

    def __init__(self, ckpt_path: Path) -> None:
        super().__init__()
        self.trunk = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            img_size=448,
            dynamic_img_size=True,
            init_values=None,
        )
        self.query = nn.Parameter(torch.zeros(1, 512))
        self.ln_q = nn.LayerNorm(512, eps=1e-6)
        self.ln_k = nn.LayerNorm(768, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, kdim=768, vdim=768, batch_first=True)
        self.ln_contrast = nn.LayerNorm(512, eps=1e-6)
        self.proj_contrast = nn.Parameter(torch.zeros(512, 512))

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        trunk_sd = {k.replace("visual.trunk.", ""): v for k, v in sd.items() if k.startswith("visual.trunk.")}
        self.trunk.load_state_dict(trunk_sd, strict=True)
        self.query.data.copy_(sd["visual.attn_pool_contrast.query"])
        self.ln_q.load_state_dict(
            {"weight": sd["visual.attn_pool_contrast.ln_q.weight"], "bias": sd["visual.attn_pool_contrast.ln_q.bias"]}
        )
        self.ln_k.load_state_dict(
            {"weight": sd["visual.attn_pool_contrast.ln_k.weight"], "bias": sd["visual.attn_pool_contrast.ln_k.bias"]}
        )
        self.attn.load_state_dict(
            {
                "q_proj_weight": sd["visual.attn_pool_contrast.attn.q_proj_weight"],
                "k_proj_weight": sd["visual.attn_pool_contrast.attn.k_proj_weight"],
                "v_proj_weight": sd["visual.attn_pool_contrast.attn.v_proj_weight"],
                "in_proj_bias": sd["visual.attn_pool_contrast.attn.in_proj_bias"],
                "out_proj.weight": sd["visual.attn_pool_contrast.attn.out_proj.weight"],
                "out_proj.bias": sd["visual.attn_pool_contrast.attn.out_proj.bias"],
            }
        )
        self.ln_contrast.load_state_dict(
            {"weight": sd["visual.ln_contrast.weight"], "bias": sd["visual.ln_contrast.bias"]}
        )
        self.proj_contrast.data.copy_(sd["visual.proj_contrast"])

        self.preprocess = T.Compose(
            [
                T.Resize(448, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop((448, 448)),
                T.ToTensor(),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def encode_image(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        tokens = self.trunk.forward_features(x)
        q = self.ln_q(self.query).unsqueeze(0).expand(x.shape[0], -1, -1)
        k = self.ln_k(tokens)
        pooled, _ = self.attn(q, k, k, need_weights=False)
        out = self.ln_contrast(pooled.squeeze(1)) @ self.proj_contrast
        if normalize:
            out = F.normalize(out, dim=-1)
        return out


class _Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Deconv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CellViT256Standalone(nn.Module):
    """Standalone CellViT-256 network matching CellViT-256-x40-AMP.pth state_dict."""

    def __init__(self, num_nuclei_classes: int = 6, num_tissue_classes: int = 19) -> None:
        super().__init__()
        self.patch_size = 16
        self.embed_dim = 384
        self.bottleneck_dim = 312
        self.encoder = HIPTViT256(num_classes=num_tissue_classes)
        self.decoder0 = nn.Sequential(_Conv2DBlock(3, 32), _Conv2DBlock(32, 64))
        self.decoder1 = nn.Sequential(_Deconv2DBlock(384, 256), _Deconv2DBlock(256, 128), _Deconv2DBlock(128, 128))
        self.decoder2 = nn.Sequential(_Deconv2DBlock(384, 256), _Deconv2DBlock(256, 256))
        self.decoder3 = nn.Sequential(_Deconv2DBlock(384, 312))
        self.nuclei_binary_map_decoder = self._make_branch(2)
        self.hv_map_decoder = self._make_branch(2)
        self.nuclei_type_maps_decoder = self._make_branch(num_nuclei_classes)

    def _make_branch(self, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", nn.ConvTranspose2d(384, 312, kernel_size=2, stride=2)),
                    (
                        "decoder3_upsampler",
                        nn.Sequential(
                            _Conv2DBlock(624, 312),
                            _Conv2DBlock(312, 312),
                            _Conv2DBlock(312, 312),
                            nn.ConvTranspose2d(312, 256, kernel_size=2, stride=2),
                        ),
                    ),
                    (
                        "decoder2_upsampler",
                        nn.Sequential(
                            _Conv2DBlock(512, 256),
                            _Conv2DBlock(256, 256),
                            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                        ),
                    ),
                    (
                        "decoder1_upsampler",
                        nn.Sequential(
                            _Conv2DBlock(256, 128),
                            _Conv2DBlock(128, 128),
                            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                        ),
                    ),
                    (
                        "decoder0_header",
                        nn.Sequential(
                            _Conv2DBlock(128, 64),
                            _Conv2DBlock(64, 64),
                            nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
                        ),
                    ),
                ]
            )
        )

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch: nn.Sequential,
    ) -> torch.Tensor:
        b4 = branch.bottleneck_upsampler(z4)
        b3 = branch.decoder3_upsampler(torch.cat([self.decoder3(z3), b4], dim=1))
        b2 = branch.decoder2_upsampler(torch.cat([self.decoder2(z2), b3], dim=1))
        b1 = branch.decoder1_upsampler(torch.cat([self.decoder1(z1), b2], dim=1))
        return branch.decoder0_header(torch.cat([self.decoder0(z0), b1], dim=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (nuclei_binary_probs [B, 2, H, W], token_map [B, 384, 16, 16])."""
        _, _, z = self.encoder.forward_with_skip_layers(x)
        z0, z1, z2, z3, z4 = x, *z
        pdim = (x.shape[-2] // 16, x.shape[-1] // 16)
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, 384, *pdim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, 384, *pdim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, 384, *pdim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, 384, *pdim)
        binary_logits = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
        return F.softmax(binary_logits, dim=1), z4


class CellToCellAttentionFusionViTS(nn.Module):
    """DINOv2 ViT-S/14 cell-to-cell attention fusion used by MLLM-HWSI."""

    def __init__(self) -> None:
        super().__init__()
        backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        self.embed_dim = 384
        self.cls_token = nn.Parameter(backbone.cls_token.detach().clone())
        self.blocks = backbone.blocks
        self.norm = backbone.norm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1).to(tokens.device, tokens.dtype)
        x = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[0, 0]


def _semantic_patch_filtering_inference(
    patches_tensor: torch.Tensor, n_diss_features: int = 32, top_k: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact MLLM-HWSI inference-mode semantic patch filtering (`semantic_patch_filtering2`)."""
    selected_patch_features: List[torch.Tensor] = []
    selected_patch_indices: List[torch.Tensor] = []

    for r in range(patches_tensor.shape[0]):
        x = patches_tensor[r]  # (256, 512)
        N = x.shape[0]
        n = min(n_diss_features, N)
        x_norm = F.normalize(x, p=2, dim=-1)
        sim = x_norm @ x_norm.T
        mean_sim = (sim.sum(dim=1) - 1.0) / max(N - 1, 1)
        seed_idx = int(torch.argmin(mean_sim).item())

        selected = [seed_idx]
        max_sim = sim[seed_idx].clone()
        max_sim[seed_idx] = float("inf")
        for _ in range(1, n):
            next_idx = int(torch.argmin(max_sim).item())
            selected.append(next_idx)
            max_sim = torch.maximum(max_sim, sim[next_idx])
            max_sim[selected] = float("inf")

        diss_idx = torch.tensor(selected, device=x.device, dtype=torch.long)
        diss_feat = x[diss_idx]
        query_feat = diss_feat[0:1]

        diss_norm = F.normalize(diss_feat, p=2, dim=-1)
        query_norm = F.normalize(query_feat, p=2, dim=-1)
        scores = (diss_norm @ query_norm.T).squeeze(1)
        _, topk_local = torch.topk(scores, k=min(top_k, diss_feat.shape[0]), largest=True)

        selected_patch_features.append(diss_feat[topk_local].cpu())
        selected_patch_indices.append(diss_idx[topk_local].cpu())

    return torch.stack(selected_patch_features, dim=0), torch.stack(selected_patch_indices, dim=0)


class MLLMHWSIFeatureExtractor:
    """Extracts and caches the 4-scale (wsi, region, patch, cell) features for MLLM-HWSI."""

    def __init__(self, paths: Optional[ProjectPaths] = None, device: str = "cuda:0") -> None:
        self.paths = paths or ProjectPaths()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. HIPT-256 & HIPT-4K
        self.hipt256 = HIPTViT256(num_classes=0)
        sd256 = torch.load(self.paths.hipt_dir / "vit_256_small_dino.pth", map_location="cpu", weights_only=False)[
            "teacher"
        ]
        sd256 = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd256.items()}
        self.hipt256.load_state_dict(sd256, strict=False)
        self.hipt256.eval().to(self.device)

        self.hipt4k = HIPTViT4K()
        sd4k = torch.load(self.paths.hipt_dir / "vit_4096_xs_dino.pth", map_location="cpu", weights_only=False)[
            "teacher"
        ]
        sd4k = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd4k.items()}
        self.hipt4k.load_state_dict(sd4k, strict=False)
        self.hipt4k.eval().to(self.device)

        self.hipt_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )

        # 2. CONCH ViT-B/16
        self.conch = ConchVisualEncoder(self.paths.conch_dir / "pytorch_model.bin").eval().to(self.device)

        # 3. CellViT-256 + DINOv2-ViTS14 cell-to-cell fusion
        self.cellvit = CellViT256Standalone(num_nuclei_classes=6, num_tissue_classes=19)
        cell_ckpt = torch.load(
            self.paths.cellvit_dir / "CellViT-256-x40-AMP.pth", map_location="cpu", weights_only=False
        )
        self.cellvit.load_state_dict(cell_ckpt["model_state_dict"], strict=True)
        self.cellvit.eval().to(self.device)
        self.cell_fusion = CellToCellAttentionFusionViTS().eval().to(self.device)

    @torch.inference_mode()
    def extract_or_load(self, wsi_path: Path, cohort: str) -> Path:
        """Return cached MLLM-HWSI .pt quadruplet path or extract and cache it in ../Datasets/."""
        out_dir = self.paths.datasets_dir / cohort / "Feat" / "MLLM_HWSI"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{wsi_path.stem}.pt"
        if out_path.is_file():
            return out_path

        region_coords = discover_region_coords_4096(wsi_path, region_size=4096)
        region_cls_list: List[torch.Tensor] = []
        region_conch_list: List[torch.Tensor] = []
        region_patch_pil: List[List[Image.Image]] = []

        use_amp = self.device.type == "cuda"
        with Image.open(wsi_path) as im:
            im_rgb = im.convert("RGB")
            for rx, ry in region_coords:
                reg_img = crop_with_padding(im_rgb, rx, ry, 4096)
                patches_pil: List[Image.Image] = []
                hipt_tensors: List[torch.Tensor] = []
                conch_tensors: List[torch.Tensor] = []
                for gy in range(16):
                    for gx in range(16):
                        p = reg_img.crop((gx * 256, gy * 256, (gx + 1) * 256, (gy + 1) * 256))
                        patches_pil.append(p)
                        hipt_tensors.append(self.hipt_transform(p))
                        conch_tensors.append(self.conch.preprocess(p))
                region_patch_pil.append(patches_pil)

                xb_hipt = torch.stack(hipt_tensors, dim=0).to(self.device)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    feat256 = self.hipt256(xb_hipt)  # (256, 384)
                    feat_grid = feat256.reshape(1, 16, 16, 384).permute(0, 3, 1, 2).contiguous()
                    cls4k = self.hipt4k(feat_grid)   # (1, 192)
                region_cls_list.append(cls4k.squeeze(0).float().cpu())

                conch_chunks: List[torch.Tensor] = []
                for b0 in range(0, 256, 64):
                    xb_c = torch.stack(conch_tensors[b0 : b0 + 64], dim=0).to(self.device)
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                        c_out = self.conch.encode_image(xb_c, normalize=False)
                    conch_chunks.append(c_out.float().cpu())
                region_conch_list.append(torch.cat(conch_chunks, dim=0))

        region_tensor = torch.stack(region_cls_list, dim=0)     # (R, 192)
        wsi_tensor = region_tensor.mean(dim=0)                  # (192,)
        all_conch = torch.stack(region_conch_list, dim=0).to(self.device)  # (R, 256, 512)

        selected_patch_feats, selected_indices = _semantic_patch_filtering_inference(
            all_conch, n_diss_features=32, top_k=16
        )  # (R, 16, 512), (R, 16)

        R, K = selected_indices.shape
        selected_cell_tensors: List[torch.Tensor] = []
        for r in range(R):
            for k_idx in range(K):
                p_idx = int(selected_indices[r, k_idx].item())
                selected_cell_tensors.append(self.hipt_transform(region_patch_pil[r][p_idx]))

        cell_encoded_list: List[torch.Tensor] = []
        for b0 in range(0, len(selected_cell_tensors), 32):
            xb_cell = torch.stack(selected_cell_tensors[b0 : b0 + 32], dim=0).to(self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                bin_probs, token_maps = self.cellvit(xb_cell)
            fg_masks = (bin_probs[:, 1] > 0.5).cpu().numpy()

            for j in range(xb_cell.shape[0]):
                tmap = token_maps[j].float()  # (384, 16, 16)
                inst_map, num_inst = connected_components(fg_masks[j])
                cell_vecs: List[torch.Tensor] = []
                for inst_id in range(1, num_inst + 1):
                    mask_bool = inst_map == inst_id
                    if mask_bool.sum() < 6:
                        continue
                    mask_t = torch.from_numpy(mask_bool.astype(np.float32))[None, None].to(self.device)
                    pooled = F.avg_pool2d(mask_t, kernel_size=16, stride=16)[0, 0] > 0
                    coords = pooled.nonzero(as_tuple=False)
                    if coords.numel() == 0:
                        ys, xs = np.where(mask_bool)
                        ty = int(np.clip(ys.mean() // 16, 0, 15))
                        tx = int(np.clip(xs.mean() // 16, 0, 15))
                        cell_vecs.append(tmap[:, ty, tx])
                    else:
                        cell_vecs.append(tmap[:, coords[:, 0], coords[:, 1]].mean(dim=1))

                if not cell_vecs:
                    cell_encoded_list.append(torch.zeros((384,), dtype=torch.float32))
                else:
                    stacked_cells = torch.stack(cell_vecs, dim=0).to(self.device)
                    fused_cell = self.cell_fusion(stacked_cells).float().cpu()
                    cell_encoded_list.append(fused_cell)

        cell_tensor = torch.stack(cell_encoded_list, dim=0).reshape(R, K, 384)
        payload = {
            "wsi": wsi_tensor.unsqueeze(0),                  # (1, 192)
            "region": region_tensor.unsqueeze(0),            # (1, R, 192)
            "patch": selected_patch_feats.unsqueeze(0),      # (1, R, 16, 512)
            "cell": cell_tensor.unsqueeze(0),                # (1, R, 16, 384)
        }
        tmp_path = out_dir / f".{wsi_path.stem}.{os.getpid()}.pt"
        torch.save(payload, tmp_path)
        tmp_path.replace(out_path)
        return out_path
