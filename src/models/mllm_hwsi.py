"""MLLM-HWSI Whole-Slide Image VLM adapter for Stage 1 closed-ended VQA."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.datasets.feature_extractor import MLLMHWSIFeatureExtractor
from src.datasets.slidebench_vqa import VQASample
from src.models.base import BaseWSIVLM
from src.utils.paths import ProjectPaths


@dataclass
class VLProjectorConfig:
    """Configuration for MLLM-HWSI's multi-level projector encoder."""

    llm_dim: int = 3584
    hidden_dim: int = 1024
    dropout: float = 0.1
    use_layernorm: bool = True
    num_query_tokens: int = 64
    projector_type: str = "mlp"


def _masked_mean(
    x: Tensor,
    mask: Optional[Tensor],
    dim: Union[int, Tuple[int, ...]],
    keepdim: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = (dim,)
    dim = tuple(sorted(dim))

    mask = mask.to(dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)

    num = x * mask
    den = mask
    for d in sorted(dim, reverse=True):
        num = num.sum(dim=d, keepdim=keepdim)
        den = den.sum(dim=d, keepdim=keepdim)

    return num / den.clamp_min(eps)


def _make_projector(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    dropout: float,
    use_layernorm: bool,
    projector_type: str = "mlp",
) -> nn.Module:
    layers: list[nn.Module] = []
    if use_layernorm:
        layers.append(nn.LayerNorm(in_dim))

    if projector_type == "linear":
        layers.append(nn.Linear(in_dim, out_dim))
    elif projector_type == "mlp":
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            ]
        )
    else:
        raise ValueError(f"Unsupported projector_type: {projector_type}")
    return nn.Sequential(*layers)


class _BaseVLProjector(nn.Module):
    def __init__(self, in_dim: int, cfg: VLProjectorConfig) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.projector = _make_projector(
            in_dim=in_dim,
            out_dim=cfg.llm_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
            projector_type=cfg.projector_type,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projector(x)


class _WSILevelProjector(nn.Module):
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 192) -> None:
        super().__init__()
        self.proj = _BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).unsqueeze(1)


class _RegionLevelProjector(nn.Module):
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 192) -> None:
        super().__init__()
        self.proj = _BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _PatchLevelProjector(nn.Module):
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 512) -> None:
        super().__init__()
        self.proj = _BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _CellLevelProjector(nn.Module):
    def __init__(self, cfg: VLProjectorConfig, in_dim: int = 384) -> None:
        super().__init__()
        self.proj = _BaseVLProjector(in_dim=in_dim, cfg=cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _RegionPatchCellFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        wsi_token: Tensor,
        region_tokens: Tensor,
        patch_tokens: Tensor,
        cell_tokens: Tensor,
        patch_mask: Optional[Tensor] = None,
        cell_mask: Optional[Tensor] = None,
    ) -> Tensor:
        pooled_patch = _masked_mean(patch_tokens, patch_mask, dim=2)
        pooled_cell = _masked_mean(cell_tokens, cell_mask, dim=2)
        wsi_expand = wsi_token.expand(-1, region_tokens.shape[1], -1)
        fused = torch.cat([wsi_expand, region_tokens, pooled_patch, pooled_cell], dim=-1)
        fused = self.fuse(fused)
        return self.out_norm(fused + region_tokens)


class _LearnedQueryResampler(nn.Module):
    def __init__(self, dim: int, num_query_tokens: int = 64, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_query_tokens, dim) / math.sqrt(dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        bsz = x.shape[0]
        q = self.queries.expand(bsz, -1, -1)
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_out, _ = self.attn(
            query=self.norm1(q),
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        z = q + attn_out
        return z + self.mlp(self.norm2(z))


class MultiLevelProjectorEncoder(nn.Module):
    """Hierarchical WSI + Region + Patch + Cell projector with 64-query resampler."""

    def __init__(self, cfg: VLProjectorConfig) -> None:
        super().__init__()
        self.wsi_proj = _WSILevelProjector(cfg, in_dim=192)
        self.region_proj = _RegionLevelProjector(cfg, in_dim=192)
        self.patch_proj = _PatchLevelProjector(cfg, in_dim=512)
        self.cell_proj = _CellLevelProjector(cfg, in_dim=384)
        self.fusion = _RegionPatchCellFusion(dim=cfg.llm_dim, dropout=cfg.dropout)
        self.compressor = _LearnedQueryResampler(
            dim=cfg.llm_dim,
            num_query_tokens=cfg.num_query_tokens,
            num_heads=8,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        wsi: Tensor,
        region: Tensor,
        patch: Tensor,
        cell: Tensor,
    ) -> Tensor:
        wsi_token = self.wsi_proj(wsi)
        region_tokens = self.region_proj(region)
        patch_tokens = self.patch_proj(patch)
        cell_tokens = self.cell_proj(cell)
        fused = self.fusion(
            wsi_token=wsi_token,
            region_tokens=region_tokens,
            patch_tokens=patch_tokens,
            cell_tokens=cell_tokens,
        )
        return self.compressor(fused)


class MLLMHWSIAdapter(BaseWSIVLM):
    """Inference adapter for MLLM-HWSI (hierarchical WSI/Region/Patch/Cell + Qwen2.5-7B)."""

    model_name: str = "mllm-hwsi"

    def __init__(
        self,
        paths: Optional[ProjectPaths] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.paths = paths or ProjectPaths()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self._feature_cache: Dict[str, Dict[str, Tensor]] = {}
        self.tokenizer = None
        self.llm = None
        self.embed_tokens = None
        self.vl_projector = None
        self._extractor: Optional[MLLMHWSIFeatureExtractor] = None

    def load(self) -> None:
        """Load MLLM-HWSI weights, multi-level projector, and tokenizer onto target device."""
        if self.llm is not None:
            return
        model_dir = self.paths.mllm_hwsi_dir
        cfg_path = model_dir / "projector_config.json"
        with cfg_path.open("r", encoding="utf-8") as f:
            proj_cfg = VLProjectorConfig(**json.load(f))

        tokenizer_source = (
            str(self.paths.qwen_dir)
            if (self.paths.qwen_dir / "tokenizer.json").is_file()
            else str(model_dir)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.llm.eval()
        self.embed_tokens = self.llm.get_input_embeddings()

        self.vl_projector = MultiLevelProjectorEncoder(proj_cfg)
        proj_payload = torch.load(model_dir / "vl_projector.pt", map_location="cpu", weights_only=False)
        proj_sd = proj_payload.get("vl_projector_state_dict", proj_payload)
        self.vl_projector.load_state_dict(proj_sd, strict=True)
        self.vl_projector.to(device=self.device, dtype=torch.float32)
        self.vl_projector.eval()

    def metadata(self) -> Dict[str, object]:
        """Return model configuration metadata for reproducibility logging."""
        return {
            "model_name": self.model_name,
            "mllm_hwsi_dir": str(self.paths.mllm_hwsi_dir),
            "encoders_dir": str(self.paths.encoders_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }


    def _get_quadruplet(self, sample: VQASample) -> Dict[str, Tensor]:
        """Load cached (wsi, region, patch, cell) tensors for `sample`, extracting if needed."""
        cache_key = f"{sample.cohort}:{sample.slide_id}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        pt_path = (
            self.paths.datasets_dir
            / sample.cohort
            / "Feat"
            / "MLLM_HWSI"
            / f"{sample.slide_id}.pt"
        )
        if not pt_path.is_file():
            if sample.wsi_path is None or not sample.wsi_path.is_file():
                raise FileNotFoundError(
                    f"Missing MLLM-HWSI features ({pt_path}) and WSI ({sample.wsi_path}) for {sample.slide_id}"
                )
            if self._extractor is None:
                self._extractor = MLLMHWSIFeatureExtractor(
                    paths=self.paths, device=str(self.device)
                )
            pt_path = self._extractor.extract_or_load(
                wsi_path=sample.wsi_path,
                cohort=sample.cohort,
            )

        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        quad = {
            k: v.to(device=self.device, dtype=torch.float32)
            for k, v in payload.items()
        }
        self._feature_cache[cache_key] = quad
        return quad

    def build_prompt(self, sample: VQASample) -> str:
        """Format a closed-ended VQA sample for MLLM-HWSI."""
        option_lines = [f"{key}. {val}" for key, val in sample.options.items()]
        options_block = "\n".join(option_lines)
        return (
            f"Question: {sample.question.strip()}\n"
            f"Options:\n{options_block}\n"
            "Select the single best option and begin your answer with the option letter (e.g., A, B, C, or D)."
        )

    def _format_chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert histopathology assistant."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            "<|im_start|>system\nYou are an expert histopathology assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @torch.inference_mode()
    def predict_vqa(self, sample: VQASample, max_new_tokens: int = 32) -> str:
        """Generate a raw closed-ended VQA response for a single sample."""
        quad = self._get_quadruplet(sample)
        slide_tokens = self.vl_projector(
            wsi=quad["wsi"],
            region=quad["region"],
            patch=quad["patch"],
            cell=quad["cell"],
        ).to(dtype=self.dtype)

        formatted_prompt = self._format_chat(self.build_prompt(sample))
        text_inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        text_attention_mask = text_inputs["attention_mask"].to(self.device)

        text_embeds = self.embed_tokens(input_ids).to(dtype=self.dtype)
        inputs_embeds = torch.cat([slide_tokens, text_embeds], dim=1)

        prefix_attention_mask = torch.ones(
            (slide_tokens.shape[0], slide_tokens.shape[1]),
            dtype=text_attention_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat([prefix_attention_mask, text_attention_mask], dim=1)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
