"""SlideChat (General-Medical-AI/SlideChat) WSI Vision-Language Model adapter."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.datasets.feature_extractor import SlideChatConchFeatureExtractor
from src.datasets.slidebench_vqa import VQASample, load_patch_features_csv
from src.models.base import BaseWSIVLM
from src.utils.paths import ProjectPaths

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


def _padding_to_multiple_of(n: int, mult: int) -> int:
    """Compute padding required so `n` becomes a multiple of `mult`."""
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class SlideChatDilatedAttention(nn.Module):
    """LongNet Dilated Multi-Head Self-Attention matching SlideChat's TorchScale implementation."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dilated_ratios: Sequence[int] = (1, 2, 4),
        segment_lengths: Sequence[int] = (512, 1024, 2048),
        layernorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dilated_ratios = tuple(dilated_ratios)
        self.segment_lengths = tuple(segment_lengths)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.inner_attn_ln = nn.LayerNorm(embed_dim, eps=layernorm_eps)

    def _dense_to_sparse(self, x: torch.Tensor, ratio: int) -> torch.Tensor:
        length = x.size(1)
        padding = _padding_to_multiple_of(length, ratio)
        head_padding = _padding_to_multiple_of(self.num_heads, ratio)

        if padding > 0 or head_padding > 0:
            x = F.pad(x, (0, 0, 0, head_padding, 0, padding), value=0.0)

        x = rearrange(x, "b (l r1) (r2 h) d -> b l h d r1 r2", r1=ratio, r2=ratio)
        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        x = rearrange(x, "b l h d r -> b l (r h) d")

        if head_padding > 0:
            x = x[:, :, : self.num_heads]
        return x

    def _sparse_to_dense(
        self, out: torch.Tensor, lse: torch.Tensor, ratio: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_padding = _padding_to_multiple_of(self.num_heads, ratio)

        if head_padding > 0:
            out = F.pad(out, (0, 0, 0, head_padding), value=0.0)
            lse = F.pad(lse, (0, 0, 0, head_padding), value=-1e8)

        out = rearrange(out, "b l (r h) d -> b l h d r", r=ratio)
        out = torch.diag_embed(out, offset=0, dim1=4, dim2=5)
        out = rearrange(out, "b l h d r1 r2 -> b (r2 h) (l r1) d", r1=ratio, r2=ratio)

        lse = rearrange(lse, "b (r h) l -> b l h r", r=ratio)
        lse = torch.diag_embed(lse, offset=0, dim1=3, dim2=4)
        lse = lse.masked_fill_(lse == 0, -1e8)
        lse = rearrange(lse, "b l h r1 r2 -> b (r2 h) (l r1) 1", r1=ratio, r2=ratio)

        if head_padding > 0:
            out = out[:, : self.num_heads]
            lse = lse[:, : self.num_heads]
        return out, lse

    def _gathering(self, x: torch.Tensor, dr: int, sl: int) -> torch.Tensor:
        curr_x = x
        seq_len = curr_x.size(1)
        sl = min(sl, seq_len)
        padding = _padding_to_multiple_of(seq_len, sl)
        if padding > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, 0, padding), value=0.0)
        curr_x = rearrange(curr_x, "b (n g) h d -> (b n) g h d", g=sl)
        curr_x = self._dense_to_sparse(curr_x, dr)
        curr_x = rearrange(curr_x, "b l h d -> (b h) l d")
        return curr_x

    def _attention_with_lse(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact scaled dot-product attention and log-sum-exp (LSE) per head."""
        q_bhld = rearrange(q, "(b h) l d -> b h l d", h=self.num_heads)
        k_bhld = rearrange(k, "(b h) l d -> b h l d", h=self.num_heads)
        v_bhld = rearrange(v, "(b h) l d -> b h l d", h=self.num_heads)

        scores = torch.matmul(q_bhld.float(), k_bhld.float().transpose(-2, -1)) * self.scaling
        lse = torch.logsumexp(scores, dim=-1)  # (b, h, l)
        probs = torch.softmax(scores, dim=-1).to(v_bhld.dtype)
        attn_out = torch.matmul(probs, v_bhld)  # (b, h, l, d)
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")
        return attn_out, lse[:, :, : attn_out.size(1)]

    def _scattering(
        self,
        outs: List[torch.Tensor],
        lses: List[torch.Tensor],
        seq_len: int,
        bsz: int,
    ) -> torch.Tensor:
        all_outs, all_lses = [], []
        for dr, o, lse in zip(self.dilated_ratios, outs, lses):
            o = rearrange(o, "b l (h d) -> b l h d", h=self.num_heads)
            o, lse = self._sparse_to_dense(o, lse, dr)
            o = rearrange(o, "(b n) h g d -> (b h) (n g) d", b=bsz)
            lse = rearrange(lse, "(b n) h g 1 -> (b h) (n g) 1", b=bsz)
            o = o[:, :seq_len]
            lse = lse[:, :seq_len]
            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0).max(0)[0]
            exp_lses = [torch.exp(lse - max_lse) for lse in all_lses]
            lse_sum = torch.stack(exp_lses, dim=0).sum(0)
            norm_lses = [lse / lse_sum for lse in exp_lses]

        out = torch.zeros_like(all_outs[0])
        for o, weight in zip(all_outs, norm_lses):
            out = out + o * weight.type_as(o)
        out = rearrange(out, "(b h) l d -> b l (h d)", h=self.num_heads)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len, _ = x.size()
        q = rearrange(self.q_proj(x), "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b l (h d) -> b l h d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b l (h d) -> b l h d", h=self.num_heads)

        outs, lses = [], []
        for sl, dr in zip(self.segment_lengths, self.dilated_ratios):
            ki = self._gathering(k, dr, sl)
            vi = self._gathering(v, dr, sl)
            qi = self._gathering(q, dr, sl)
            out, lse = self._attention_with_lse(qi, ki, vi)
            outs.append(out)
            lses.append(lse)

        attn = self._scattering(outs, lses, tgt_len, bsz)
        attn = self.inner_attn_ln(attn)
        return self.out_proj(attn)


class SlideChatFFN(nn.Module):
    """SubLN FeedForwardNetwork matching TorchScale's FeedForwardNetwork."""

    def __init__(
        self, embed_dim: int = 512, ffn_dim: int = 1024, layernorm_eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.ffn_layernorm(x)
        return self.fc2(x)


class SlideChatLongNetLayer(nn.Module):
    """Single LongNet encoder layer (`LongNet_2_layers_512_dim`)."""

    def __init__(self, embed_dim: int = 512, ffn_dim: int = 1024, num_heads: int = 8) -> None:
        super().__init__()
        self.self_attn = SlideChatDilatedAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ffn = SlideChatFFN(embed_dim=embed_dim, ffn_dim=ffn_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = residual + self.self_attn(x)

        residual = x
        x = self.final_layer_norm(x)
        x = residual + self.ffn(x)
        return x


class SlideChatLongNetEncoder(nn.Module):
    """2-layer 512-dim LongNet slide encoder used by SlideChat."""

    def __init__(self, num_layers: int = 2, embed_dim: int = 512, ffn_dim: int = 1024) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SlideChatLongNetLayer(embed_dim=embed_dim, ffn_dim=ffn_dim) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(
        self, src_tokens: Optional[torch.Tensor] = None, token_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        assert token_embeddings is not None
        x = token_embeddings
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return {"encoder_out": x}


class SlideChatProjector(nn.Module):
    """2-layer MLP visual-language projector mapping 512-d slide tokens to 3584-d Qwen2.5 tokens."""

    def __init__(self, visual_hidden_size: int = 512, llm_hidden_size: int = 3584) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(visual_hidden_size, llm_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SlideChatAdapter(BaseWSIVLM):
    """Inference adapter for SlideChat (General-Medical-AI/SlideChat_Weight)."""

    model_name = "SlideChat"

    def __init__(
        self,
        slidechat_ckpt_path: Path,
        qwen_dir: Path,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_patches: int = 38400,
        paths: Optional[ProjectPaths] = None,
    ) -> None:
        self.slidechat_ckpt_path = slidechat_ckpt_path
        self.qwen_dir = qwen_dir
        self.paths = paths or ProjectPaths()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        if self.device.type == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            if total_vram_gb <= 26.0 and max_patches > 12288:
                max_patches = 12288
        self.max_patches = max_patches

        self.tokenizer: Optional[Any] = None
        self.llm: Optional[Any] = None
        self.longnet_encoder: Optional[SlideChatLongNetEncoder] = None
        self.projector: Optional[SlideChatProjector] = None
        self._conch_extractor: Optional[SlideChatConchFeatureExtractor] = None
        self._last_cache_key: Optional[str] = None
        self._last_pixel_values: Optional[torch.Tensor] = None

    def load(self) -> None:
        """Load Qwen2.5-7B-Instruct and apply SlideChat Stage-2 weights (LLM + LongNet + Projector)."""
        if self.llm is not None:
            return

        print(f"[SlideChat] Loading tokenizer and base LLM config from {self.qwen_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.qwen_dir), trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            str(self.qwen_dir),
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )

        self.longnet_encoder = SlideChatLongNetEncoder(
            num_layers=2, embed_dim=512, ffn_dim=1024
        )
        self.projector = SlideChatProjector(visual_hidden_size=512, llm_hidden_size=3584)

        ckpt_file = self.slidechat_ckpt_path
        if ckpt_file.is_dir():
            ckpt_file = ckpt_file / "stage2_pth" / "mp_rank_00_model_states.pt"

        print(f"[SlideChat] Loading Stage-2 checkpoint from {ckpt_file}...")
        raw_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        module_sd: Dict[str, torch.Tensor] = raw_ckpt.get("module", raw_ckpt)

        longnet_sd = {
            k[len("LongNet_encoder.") :]: v
            for k, v in module_sd.items()
            if k.startswith("LongNet_encoder.")
        }
        projector_sd = {
            k[len("projector.") :]: v
            for k, v in module_sd.items()
            if k.startswith("projector.")
        }
        llm_sd = {
            k[len("llm.") :]: v
            for k, v in module_sd.items()
            if k.startswith("llm.")
        }

        self.longnet_encoder.load_state_dict(longnet_sd, strict=True)
        self.projector.load_state_dict(projector_sd, strict=True)
        self.llm.load_state_dict(llm_sd, strict=True)

        del raw_ckpt, module_sd, longnet_sd, projector_sd, llm_sd

        self.longnet_encoder = self.longnet_encoder.to(device=self.device, dtype=self.dtype).eval()
        self.projector = self.projector.to(device=self.device, dtype=self.dtype).eval()
        self.llm = self.llm.to(device=self.device, dtype=self.dtype).eval()
        print(f"[SlideChat] Successfully loaded on {self.device} ({self.dtype}).")

    def _prepare_multimodal_inputs(
        self, prompt_text: str, pixel_values: torch.Tensor
    ) -> Dict[str, Any]:
        """Construct multimodal inputs_embeds, position_ids, and attention_mask for Qwen2.5."""
        assert self.tokenizer is not None and self.llm is not None

        formatted = (
            f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        chunks = formatted.split(DEFAULT_IMAGE_TOKEN)
        assert len(chunks) == 2

        chunk0_ids = self.tokenizer.encode(chunks[0])
        chunk1_ids = self.tokenizer.encode(chunks[1], add_special_tokens=False)

        embed_layer = self.llm.get_input_embeddings()
        embeds_0 = embed_layer(torch.tensor(chunk0_ids, device=self.device))
        embeds_1 = embed_layer(torch.tensor(chunk1_ids, device=self.device))

        vis_embeds = pixel_values[0]  # (num_patches, llm_hidden_size)
        combined_embeds = torch.cat([embeds_0, vis_embeds, embeds_1], dim=0).unsqueeze(0)
        seq_len = combined_embeds.size(1)

        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        return {
            "inputs_embeds": combined_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    @torch.inference_mode()
    def predict_vqa(self, sample: VQASample, max_new_tokens: int = 128) -> str:
        """Run SlideChat inference on a single VQASample."""
        self.load()
        assert (
            self.longnet_encoder is not None
            and self.projector is not None
            and self.llm is not None
            and self.tokenizer is not None
        )

        cache_key = f"{sample.cohort}:{sample.slide_id}"
        if self._last_cache_key == cache_key and self._last_pixel_values is not None:
            pixel_values = self._last_pixel_values
        else:
            self._last_pixel_values = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            feat_path = sample.feature_path
            if feat_path is None or not feat_path.exists():
                if sample.wsi_path is None or not sample.wsi_path.is_file():
                    raise FileNotFoundError(
                        f"SlideChat requires patch features or WSI for slide {sample.slide_id}, "
                        f"but feature_path={sample.feature_path} and wsi_path={sample.wsi_path} were not found."
                    )
                if self._conch_extractor is None:
                    self._conch_extractor = SlideChatConchFeatureExtractor(
                        paths=self.paths, device=str(self.device)
                    )
                feat_path = self._conch_extractor.extract_or_load(
                    wsi_path=sample.wsi_path,
                    cohort=sample.cohort,
                )

            patch_feats = load_patch_features_csv(
                feat_path, max_patches=self.max_patches
            ).to(device=self.device, dtype=self.dtype)

            # SlideChat LongNet expects (N, 1, 512) -> permute back to (1, N, 512)
            encoded = self.longnet_encoder(
                src_tokens=None, token_embeddings=patch_feats.permute(1, 0, 2)
            )["encoder_out"].permute(1, 0, 2)
            pixel_values = self.projector(encoded)
            del patch_feats, encoded
            self._last_cache_key = cache_key
            self._last_pixel_values = pixel_values

        prompt_text = sample.format_closed_prompt()
        mm_inputs = self._prepare_multimodal_inputs(prompt_text, pixel_values)

        eos_id = self.tokenizer.eos_token_id
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [tid for tid in (eos_id, im_end_id) if isinstance(tid, int) and tid >= 0]

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids if eos_ids else eos_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else eos_id
            ),
        )
        output_ids = self.llm.generate(**mm_inputs, generation_config=gen_config)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if decoded.endswith("<|im_end|>"):
            decoded = decoded[: -len("<|im_end|>")].strip()
        return decoded

    def metadata(self) -> Dict[str, Any]:
        """Return SlideChat model configuration metadata."""
        return {
            "model_name": self.model_name,
            "checkpoint_path": str(self.slidechat_ckpt_path),
            "base_llm_path": str(self.qwen_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "max_patches": self.max_patches,
        }
