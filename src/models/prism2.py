"""PRISM2 Whole-Slide Image VLM adapter for Stage 1 closed-ended VQA."""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Dict, Optional

from safetensors.torch import load_file
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.phi3.modeling_phi3 import Phi3Model

from src.datasets.feature_extractor import Virchow2FeatureExtractor
from src.datasets.slidebench_vqa import VQASample
from src.models.base import BaseWSIVLM
from src.utils.paths import ProjectPaths


def _sdpa_flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
) -> Tensor:
    """Drop-in PyTorch SDPA replacement for `flash_attn_varlen_func` with GQA support."""
    batch_size = cu_seqlens_q.numel() - 1
    outputs: list[Tensor] = []
    enable_gqa = q.shape[1] != k.shape[1]
    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        k_start = int(cu_seqlens_k[b].item())
        k_end = int(cu_seqlens_k[b + 1].item())
        q_b = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_b = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_b = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        out_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b, is_causal=causal, enable_gqa=enable_gqa
        )
        outputs.append(out_b.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _install_prism2_runtime_shims() -> None:
    """Ensure flash_attn_varlen_func and Phi3 4D causal mask helper exist at runtime."""
    if "flash_attn" not in sys.modules:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            mod = types.ModuleType("flash_attn")
            mod.flash_attn_varlen_func = _sdpa_flash_attn_varlen_func  # type: ignore[attr-defined]
            sys.modules["flash_attn"] = mod

    if not hasattr(Phi3Model, "_prepare_4d_causal_attention_mask_with_cache_position"):

        @staticmethod
        def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            cache_position: Tensor,
            batch_size: int,
            device: torch.device,
            config,
            past_key_values=None,
        ) -> Tensor:
            dummy_embeds = torch.empty(
                (batch_size, sequence_length, config.hidden_size),
                dtype=dtype,
                device=device,
            )
            bool_mask = create_causal_mask(
                config=config,
                inputs_embeds=dummy_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=cache_position.unsqueeze(0),
                allow_is_causal_skip=False,
            )
            # Convert boolean mask (True=attend, False=mask) to additive float mask (0.0=attend, min=mask)
            # so PRISM2's `attn_mask[i, :, first:last, first:last] = 0` enables bidirectional image attention.
            min_val = torch.finfo(dtype).min
            float_mask = torch.where(
                bool_mask,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), min_val, dtype=dtype, device=device),
            )
            return float_mask

        Phi3Model._prepare_4d_causal_attention_mask_with_cache_position = (  # type: ignore[attr-defined]
            _prepare_4d_causal_attention_mask_with_cache_position
        )


def _load_prism2_modules(prism2_dir):
    """Dynamically load configuration_prism2, processing_prism2, and modeling_prism2 from local disk."""
    _install_prism2_runtime_shims()
    prism2_dir_str = str(prism2_dir)
    if prism2_dir_str not in sys.path:
        sys.path.insert(0, prism2_dir_str)

    for mod_name in ("configuration_prism2", "processing_prism2", "modeling_prism2"):
        if mod_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(mod_name, prism2_dir / f"{mod_name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

    return (
        sys.modules["configuration_prism2"].Prism2Config,
        sys.modules["processing_prism2"].Prism2Processor,
        sys.modules["modeling_prism2"].Prism2Model,
    )


class PRISM2Adapter(BaseWSIVLM):
    """Inference adapter for Paige PRISM2 (Virchow2 tile embeddings + Perceiver + Phi-3)."""

    model_name: str = "prism2"

    def __init__(
        self,
        paths: Optional[ProjectPaths] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.paths = paths or ProjectPaths()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self._feature_cache: Dict[str, Tensor] = {}
        self.tokenizer = None
        self.processor = None
        self.model = None
        self._virchow2_extractor: Optional[Virchow2FeatureExtractor] = None

    def load(self) -> None:
        """Load PRISM2 weights, processor, and tokenizer onto target device."""
        if self.model is not None:
            return
        Prism2Config, Prism2Processor, Prism2Model = _load_prism2_modules(self.paths.prism2_dir)

        config = Prism2Config.from_pretrained(str(self.paths.prism2_dir), trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.paths.prism2_dir), trust_remote_code=True)
        self.processor = Prism2Processor(
            tokenizer=self.tokenizer,
            num_img_tokens=config.num_img_tokens,
        )

        with torch.device(self.device):
            torch_default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                model = Prism2Model(config)
            finally:
                torch.set_default_dtype(torch_default_dtype)

        for shard_path in sorted(self.paths.prism2_dir.glob("model-*.safetensors")):
            shard_sd = load_file(str(shard_path), device=str(self.device))
            model.load_state_dict(shard_sd, strict=False, assign=True)

        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model._tokenizer = self.tokenizer
        self.model.eval()

    def metadata(self) -> Dict[str, object]:
        """Return model configuration metadata for reproducibility logging."""
        return {
            "model_name": self.model_name,
            "prism2_dir": str(self.paths.prism2_dir),
            "virchow2_dir": str(self.paths.virchow2_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }


    def _get_tile_embeddings(self, sample: VQASample) -> Tensor:
        """Load cached (N, 1280) Virchow2 tile embeddings for `sample`, extracting if needed."""
        cache_key = f"{sample.cohort}:{sample.slide_id}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        pt_path = (
            self.paths.datasets_dir
            / sample.cohort
            / "Feat"
            / "PRISM2_Virchow2"
            / f"{sample.slide_id.replace('/', '_')}.pt"
        )
        if not pt_path.is_file():
            if sample.wsi_path is None or not sample.wsi_path.is_file():
                raise FileNotFoundError(
                    f"Missing Virchow2 features ({pt_path}) and WSI ({sample.wsi_path}) for {sample.slide_id}"
                )
            if self._virchow2_extractor is None:
                self._virchow2_extractor = Virchow2FeatureExtractor(
                    paths=self.paths, device=str(self.device)
                )
            pt_path = self._virchow2_extractor.extract_or_load(
                wsi_path=sample.wsi_path,
                cohort=sample.cohort,
                feature_csv_path=sample.feature_path,
            )

        tiles = torch.load(pt_path, map_location="cpu", weights_only=False).to(
            device=self.device, dtype=self.dtype
        )
        self._feature_cache[cache_key] = tiles
        return tiles

    def build_prompt(self, sample: VQASample) -> str:
        """Format a closed-ended VQA sample following PRISM2's multiple-choice convention."""
        option_lines = [f"{key}. {val}" for key, val in sample.options.items()]
        options_block = "\n".join(option_lines)
        return (
            f"{sample.question.strip()}\n"
            f"Options:\n{options_block}\n"
            "Answer with the single best option letter (e.g., A, B, C, or D)."
        )

    @torch.inference_mode()
    def predict_vqa(self, sample: VQASample, max_new_tokens: int = 32) -> str:
        """Generate a raw closed-ended VQA response for a single sample."""
        tiles = self._get_tile_embeddings(sample)
        batch = self.processor([tiles])
        tile_embeddings = batch["tile_embeddings"].to(device=self.device, dtype=self.dtype)
        attention_mask = batch["attention_mask"].to(device=self.device)

        prompt = self.build_prompt(sample)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=(self.device.type == "cuda"),
        ):
            responses = self.model.get_response(
                tile_embeddings=tile_embeddings,
                attention_mask=attention_mask,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            response_text = responses[0].strip()

            # For Yes/No binary questions, if free-text generation does not clearly start with A/B/Yes/No,
            # use PRISM2's calibrated `yes_no_score` log-prob head as fallback.
            if sample.question_type == "binary_yes_no" and not response_text:
                score = float(
                    self.model.yes_no_score(
                        tile_embeddings=tile_embeddings,
                        attention_mask=attention_mask,
                        question=sample.question,
                    )[0].item()
                )
                return "A. Yes" if score >= 0.5 else "B. No"

        return response_text
