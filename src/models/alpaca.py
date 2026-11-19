"""ALPaCA (CNX-PathLLM/Llama-slideQA / ZeyuGaoAi/ALPaCA) WSI VLM adapter."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.datasets.feature_extractor import (
    ALPACA_SCALE_ORDER,
    ALPACA_SCALE_TO_GMM_KEY,
    ALPaCAConchFeatureExtractor,
)
from src.datasets.slidebench_caption import CaptionSample
from src.datasets.slidebench_vqa import VQASample
from src.models.base import BaseWSIVLM
from src.models.slidechat import SlideChatDilatedAttention, SlideChatFFN
from src.utils.paths import ProjectPaths

_CLOSE_MC_PROMPT = (
    " Please provide only the answer (for example, A, B, etc.) "
    "for the following question. Do not include any explanations or "
    "additional text. Just give the letter."
)
_CLOSE_TF_PROMPT = (
    " Please provide only the answer (either Yes or No) for the "
    "following statement. Do not include any explanations or "
    "additional text. Just give Yes or No."
)


def _optimal_segment_and_dilation(
    max_wsi_size: int = 262144, tile_size: int = 1024
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Compute LongNet segment_lengths and dilated_ratios matching `LongNetViT.get_optimal_segment_length`."""
    max_seq_len = (max_wsi_size // tile_size) ** 2
    seg_log2 = np.linspace(np.log2(1024), int(np.log2(max_seq_len)), 5)
    segment_lengths = tuple(int(x) for x in np.power(2, seg_log2).astype(int).tolist())
    dilated_ratios = tuple(2 ** i for i in range(len(segment_lengths)))
    return segment_lengths, dilated_ratios


class _ALPaCALongNetLayer(nn.Module):
    """Single 512-dim LongNet encoder layer (`ffn_dim=2048`) matching `LongNet_1_layers_512_dim`."""

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_dim: int = 2048,
        num_heads: int = 8,
        dilated_ratios: Sequence[int] = (1, 2, 4, 8, 16),
        segment_lengths: Sequence[int] = (1024, 2048, 4096, 8192, 16384),
    ) -> None:
        super().__init__()
        self.self_attn = SlideChatDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dilated_ratios=dilated_ratios,
            segment_lengths=segment_lengths,
            layernorm_eps=1e-5,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ffn = SlideChatFFN(embed_dim=embed_dim, ffn_dim=ffn_dim, layernorm_eps=1e-5)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = residual + self.self_attn(x)

        residual = x
        x = self.final_layer_norm(x)
        x = residual + self.ffn(x)
        return x


class _ALPaCALongNetSubEncoder(nn.Module):
    """1-layer LongNet sub-encoder inside `LongNetViT.encoder_wsi`."""

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_dim: int = 2048,
        num_heads: int = 8,
        dilated_ratios: Sequence[int] = (1, 2, 4, 8, 16),
        segment_lengths: Sequence[int] = (1024, 2048, 4096, 8192, 16384),
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ALPaCALongNetLayer(
                    embed_dim=embed_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dilated_ratios=dilated_ratios,
                    segment_lengths=segment_lengths,
                )
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(
        self,
        src_tokens: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert token_embeddings is not None
        x = token_embeddings
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return {"encoder_out": x}


class _ALPaCACrossAttention(nn.Module):
    """Wrapper around `nn.MultiheadAttention` matching `gigapath.slide_encoder.CrossAttention`."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn(query, key, value, key_padding_mask=key_padding_mask)


class ALPaCALongNetViT(nn.Module):
    """Vision-Text Interaction LongNetViT adaptor for a single magnification scale (`level 0/1/2`)."""

    def __init__(
        self,
        embed_dim: int = 512,
        slide_ngrids: int = 1000,
        tile_size: int = 1024,
        max_wsi_size: int = 262144,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids ** 2
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.register_buffer("pos_embed", torch.zeros(1, num_patches, embed_dim), persistent=True)
        self.query_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.instruct_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        segment_lengths, dilated_ratios = _optimal_segment_and_dilation(
            max_wsi_size=max_wsi_size, tile_size=tile_size
        )
        self.encoder_wsi = nn.ModuleList(
            [
                _ALPaCALongNetSubEncoder(
                    embed_dim=embed_dim,
                    ffn_dim=2048,
                    num_heads=num_heads,
                    dilated_ratios=dilated_ratios,
                    segment_lengths=segment_lengths,
                )
                for _ in range(num_layers)
            ]
        )
        self.self_attention = nn.ModuleList(
            [_ALPaCACrossAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.cross_attention = nn.ModuleList(
            [_ALPaCACrossAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(num_layers)])

    def coords_to_pos(self, coords: torch.Tensor, patch_size: float = 1024.0) -> torch.Tensor:
        """Convert `(N, L, 2)` patch coordinates into 1D `pos_embed` indices `(N, L)`."""
        coords_grid = torch.floor(coords.float() / float(patch_size))
        pos = coords_grid[..., 0] * self.slide_ngrids + coords_grid[..., 1]
        return pos.long().clamp(0, self.pos_embed.size(1) - 1)

    def forward(
        self,
        querys: torch.Tensor,
        contexts: torch.Tensor,
        instructs: torch.Tensor,
        coords: torch.Tensor,
        patch_size: float = 1024.0,
        self_attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.pos_embed.dtype
        querys = querys.to(target_dtype)
        contexts = contexts.to(target_dtype)
        instructs = instructs.to(target_dtype)
        query_length = querys.shape[1]

        if self_attention_mask is not None:
            query_mask_ext = torch.ones(
                (querys.shape[0], querys.shape[1]),
                dtype=self_attention_mask.dtype,
                device=self_attention_mask.device,
            )
            self_attention_mask = torch.cat((query_mask_ext, self_attention_mask), dim=-1)

        querys = self.query_norm(querys)
        instructs = self.instruct_norm(instructs)
        pos = self.coords_to_pos(coords=coords, patch_size=patch_size)
        contexts = contexts * math.sqrt(self.embed_dim) + self.pos_embed[:, pos, :].squeeze(0)

        query_pic_weights = torch.empty(0, device=querys.device)
        for i in range(self.num_layers):
            contexts = self.encoder_wsi[i](
                src_tokens=None,
                token_embeddings=contexts,
                encoder_padding_mask=key_padding_mask,
            )["encoder_out"]
            combined_querys = torch.cat((querys, instructs), dim=1)
            self_attn_out, _ = self.self_attention[i](
                query=combined_querys,
                key=combined_querys,
                value=combined_querys,
                key_padding_mask=(
                    ~self_attention_mask.bool() if self_attention_mask is not None else None
                ),
            )
            querys = self_attn_out[:, :query_length, :]
            querys = self.norm[i](querys)
            querys, query_pic_weights = self.cross_attention[i](
                query=querys,
                key=contexts,
                value=contexts,
                key_padding_mask=key_padding_mask,
            )
        return querys, query_pic_weights


class _ALPaCAReducer(nn.Module):
    """Linear reducer projecting LLM hidden dimension (4096) to slide embed dimension (512)."""

    def __init__(self, in_chans: int = 4096, embed_dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=True)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class ALPaCAHybridAdaptor(nn.Module):
    """Hybrid LongNet + GMM multi-scale vision-language adaptor matching `WPathVLM` non-LLM state_dict."""

    def __init__(
        self,
        llm_hidden_size: int = 4096,
        embed_dim: int = 512,
        n_heads: Sequence[int] = (32, 16, 8),
    ) -> None:
        super().__init__()
        self.n_heads = list(n_heads)
        self.n_level = len(self.n_heads)
        self.embed_dim = embed_dim

        self.reduce = _ALPaCAReducer(in_chans=llm_hidden_size, embed_dim=embed_dim)
        self.query = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.n_heads[i], embed_dim)) for i in range(self.n_level)]
        )
        tile_sizes = (1024, 2048, 4096)
        self.longnet_encoder_list = nn.ModuleList(
            [
                ALPaCALongNetViT(
                    embed_dim=embed_dim,
                    slide_ngrids=1000,
                    tile_size=tile_sizes[level],
                    max_wsi_size=262144,
                    num_layers=2,
                    num_heads=8,
                )
                for level in range(self.n_level)
            ]
        )
        self.resampler_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, llm_hidden_size, bias=False),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.query_gmm = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.n_heads[i], embed_dim)) for i in range(self.n_level)]
        )
        self.query_norm_gmm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.downsampler_gmm = nn.Sequential(
            nn.LayerNorm(embed_dim * 2 + 1),
            nn.Linear(embed_dim * 2 + 1, embed_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.gmm_xatt_list = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
                for _ in range(self.n_level)
            ]
        )
        self.resampler_layer_gmm = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, llm_hidden_size, bias=False),
            nn.ReLU(),
            nn.Dropout(0.25),
        )


class ALPaCAAdapter(BaseWSIVLM):
    """Unified inference adapter for ALPaCA (`CNX-PathLLM/Llama-slideQA` on `Llama-3.1-8B-Instruct`)."""

    model_name = "alpaca"

    def __init__(
        self,
        paths: Optional[ProjectPaths] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        ckpt_filename: str = "Llama-slideQA.bin",
    ) -> None:
        self.paths = paths or ProjectPaths()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.ckpt_path = self.paths.alpaca_dir / ckpt_filename

        self.tokenizer: Optional[Any] = None
        self.llm: Optional[Any] = None
        self.embedding_layer: Optional[nn.Module] = None
        self.adaptor: Optional[ALPaCAHybridAdaptor] = None
        self.image_token_id: List[int] = []
        self._extractor: Optional[ALPaCAConchFeatureExtractor] = None
        self._feature_cache: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        """Load Llama-3.1-8B-Instruct and stream ALPaCA `Llama-slideQA.bin` weights via `mmap=True`."""
        if self.llm is not None and self.adaptor is not None:
            return

        print(f"[ALPaCA] Loading tokenizer and Llama-3.1-8B config from {self.paths.llama3_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.paths.llama3_dir))
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        new_tokens = [
            "<|Question|>",
            "<|Prompt|>",
            "<|Answer|>",
            "<|Image|>",
            "<|High|>",
            "<|Mid|>",
            "<|Low|>",
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        new_token_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
        self.image_token_id = list(new_token_ids[3:])  # [<|Image|>, <|High|>, <|Mid|>, <|Low|>]

        # Build LLM architecture directly in target dtype and move to GPU first to avoid CPU RAM duplication
        llm_config = AutoConfig.from_pretrained(str(self.paths.llama3_dir))
        llm_config.use_cache = True
        self.llm = AutoModelForCausalLM.from_config(
            llm_config,
            torch_dtype=self.dtype,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        self.llm.to(device=self.device, dtype=self.dtype).eval()
        gc.collect()

        self.adaptor = ALPaCAHybridAdaptor(
            llm_hidden_size=self.llm.config.hidden_size,
            embed_dim=512,
            n_heads=(32, 16, 8),
        )

        print(f"[ALPaCA] Streaming WPathVLM checkpoint (mmap=True) from {self.ckpt_path}...")
        raw_sd = torch.load(
            self.ckpt_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        # Stream LLM weights tensor-by-tensor directly onto target device to keep peak CPU RAM < 3 GB
        llm_params: Dict[str, torch.Tensor] = dict(self.llm.named_parameters())
        llm_buffers: Dict[str, torch.Tensor] = dict(self.llm.named_buffers())
        loaded_llm_keys: Set[str] = set()
        adaptor_sd: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for key in list(raw_sd.keys()):
                val = raw_sd.pop(key)
                if key.startswith("llm."):
                    sub_key = key[len("llm.") :]
                    target = llm_params.get(sub_key)
                    if target is None:
                        target = llm_buffers.get(sub_key)
                    if target is None:
                        raise KeyError(f"Unexpected LLM key in checkpoint: {key}")
                    target.copy_(val.to(device=self.device, dtype=target.dtype))
                    loaded_llm_keys.add(sub_key)
                elif key == "embedding_layer.weight":
                    sub_key = "model.embed_tokens.weight"
                    if sub_key not in loaded_llm_keys:
                        target = llm_params[sub_key]
                        target.copy_(val.to(device=self.device, dtype=target.dtype))
                        loaded_llm_keys.add(sub_key)
                else:
                    adaptor_sd[key] = val
                del val

        missing_llm = set(llm_params.keys()) - loaded_llm_keys
        if missing_llm:
            raise RuntimeError(f"Missing LLM parameters in {self.ckpt_path}: {sorted(missing_llm)[:5]}")

        self.adaptor.load_state_dict(adaptor_sd, strict=True)
        del raw_sd, adaptor_sd, llm_params, llm_buffers
        gc.collect()

        # Match upstream WPathVLM precision: LongNet in bfloat16, GMM/resamplers in float32, LLM in self.dtype
        self.adaptor.to(device=self.device, dtype=torch.float32).eval()
        self.adaptor.longnet_encoder_list.to(device=self.device, dtype=self.dtype).eval()
        self.embedding_layer = self.llm.get_input_embeddings()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[ALPaCA] Successfully loaded on {self.device} (LLM={self.dtype}, strict=True, mmap=True).")

    @staticmethod
    def format_prompt(question: str, mode: str = "open") -> str:
        """Format prompt text with ALPaCA special tokens (`open`, `close-mc`, or `close-tf`)."""
        if mode == "open":
            return f"<|Question|>{question}<|Answer|>"
        if mode == "close-mc":
            return f"<|Question|>{question}<|Prompt|>{_CLOSE_MC_PROMPT}<|Answer|>"
        if mode == "close-tf":
            return f"<|Question|>{question}<|Prompt|>{_CLOSE_TF_PROMPT}<|Answer|>"
        raise ValueError(f"Unsupported ALPaCA prompt mode: {mode!r}")

    def _get_slide_features(
        self,
        slide_id: str,
        cohort: str,
        wsi_path: Optional[Path] = None,
        feature_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Retrieve or extract 3-scale LongNet + GMM features for `slide_id`."""
        cache_key = f"{cohort}:{slide_id}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        if self._extractor is None:
            self._extractor = ALPaCAConchFeatureExtractor(
                paths=self.paths, device=str(self.device)
            )
        payload = self._extractor.extract_or_load(
            slide_id=slide_id,
            cohort=cohort,
            wsi_path=wsi_path,
            feature_path=feature_path,
        )
        if len(self._feature_cache) >= 8:
            oldest = next(iter(self._feature_cache))
            del self._feature_cache[oldest]
        self._feature_cache[cache_key] = payload
        return payload

    def _compute_fusion_embeddings(
        self,
        question: str,
        mode: str,
        slide_payload: Dict[str, Any],
        max_seq_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run 3-scale LongNet + GMM hybrid aggregation and construct LLM `inputs_embeds`."""
        assert (
            self.tokenizer is not None
            and self.llm is not None
            and self.embedding_layer is not None
            and self.adaptor is not None
        )

        prompt_text = self.format_prompt(question, mode=mode)
        enc = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
        )
        enc_q = self.tokenizer(
            question,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
        )

        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=self.device)
        text_attention_mask = torch.tensor(
            [enc["attention_mask"]], dtype=torch.long, device=self.device
        )
        # Strip leading BOS token from instruction stream (matches MyDataCollatorForWPathVLM)
        instruct_ids_list = enc_q["input_ids"][1:] or enc_q["input_ids"]
        instruct_mask_list = enc_q["attention_mask"][1:] or enc_q["attention_mask"]
        input_ids_instruct = torch.tensor(
            [instruct_ids_list], dtype=torch.long, device=self.device
        )
        attention_mask_instruct = torch.tensor(
            [instruct_mask_list], dtype=torch.long, device=self.device
        )

        instruct_embs = self.embedding_layer(input_ids_instruct).float()
        instruct_embs = self.adaptor.reduce(instruct_embs)

        patch_size_map = {0: 1024.0, 1: 2048.0, 2: 4096.0}
        gmm_keys = ("f1024", "f2048", "f4096")

        agged_wsi_embs: List[torch.Tensor] = []
        for level in range(self.adaptor.n_level):
            # 1. LongNet branch
            fea_l = slide_payload["longnet"][f"fea{level}"].unsqueeze(0).to(
                device=self.device, dtype=torch.float32
            )
            cor_l = slide_payload["longnet"][f"cor{level}"].unsqueeze(0).to(
                device=self.device, dtype=torch.long
            )
            mask_l = torch.ones((1, fea_l.size(1)), dtype=torch.bool, device=self.device)
            query_l = self.adaptor.query[level].repeat(1, 1, 1)

            longnet_out, _ = self.adaptor.longnet_encoder_list[level](
                querys=query_l,
                contexts=fea_l,
                instructs=instruct_embs,
                coords=cor_l,
                patch_size=patch_size_map[level],
                self_attention_mask=attention_mask_instruct,
                key_padding_mask=~mask_l,
            )
            longnet_resampled = self.adaptor.resampler_layer(longnet_out.float())

            # 2. GMM branch
            gmm_tok = slide_payload["gmm"][gmm_keys[level]].unsqueeze(0).to(
                device=self.device, dtype=torch.float32
            )
            query_g = self.adaptor.query_gmm[level].repeat(1, 1, 1)
            gmm_down = self.adaptor.downsampler_gmm(gmm_tok)
            gmm_out, _ = self.adaptor.gmm_xatt_list[level](
                self.adaptor.query_norm_gmm(query_g), gmm_down, gmm_down
            )
            gmm_resampled = self.adaptor.resampler_layer_gmm(gmm_out.float())

            agged_wsi_embs.append((longnet_resampled + gmm_resampled).to(dtype=self.dtype))

        # Hierarchical fusion: <|Image|> + [<|High|> + L0] + [<|Mid|> + L1] + [<|Low|> + L2] + text
        text_token_embs = self.embedding_layer(input_ids).to(dtype=self.dtype)
        img_tok_tensor = torch.tensor(self.image_token_id[0], device=self.device)
        batch_img_tok_emb = self.embedding_layer(img_tok_tensor).to(dtype=self.dtype).view(1, 1, -1)

        mag_wsi_embs: List[torch.Tensor] = []
        for i in range(self.adaptor.n_level):
            mag_tok_tensor = torch.tensor(self.image_token_id[i + 1], device=self.device)
            batch_mag_emb = self.embedding_layer(mag_tok_tensor).to(dtype=self.dtype).view(1, 1, -1)
            mag_wsi_embs.append(torch.cat((batch_mag_emb, agged_wsi_embs[i]), dim=1))

        all_mag_embs = torch.cat(mag_wsi_embs, dim=1)
        fusion_embs = torch.cat((batch_img_tok_emb, all_mag_embs, text_token_embs), dim=1)

        pad_len = fusion_embs.size(1) - text_attention_mask.size(1)
        prefix_mask = torch.ones(
            (1, pad_len), dtype=text_attention_mask.dtype, device=self.device
        )
        full_attention_mask = torch.cat((prefix_mask, text_attention_mask), dim=1)
        return fusion_embs, full_attention_mask

    @torch.inference_mode()
    def generate_answer(
        self,
        question: str,
        mode: str,
        slide_payload: Dict[str, Any],
        max_new_tokens: int = 128,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate an answer string given a question, prompt mode, and 3-scale slide payload."""
        self.load()
        assert self.llm is not None and self.tokenizer is not None

        fusion_embs, attention_mask = self._compute_fusion_embeddings(
            question=question, mode=mode, slide_payload=slide_payload
        )
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": 1.0, "top_k": 50, "top_p": 0.95})

        gen_config = GenerationConfig(**gen_kwargs)
        output_ids = self.llm.generate(
            inputs_embeds=fusion_embs,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def predict_vqa(self, sample: VQASample, max_new_tokens: int = 64) -> str:
        """Run ALPaCA closed-ended VQA inference (`close-tf` for Yes/No, `close-mc` for A/B/C/D)."""
        slide_payload = self._get_slide_features(
            slide_id=sample.slide_id,
            cohort=sample.cohort,
            wsi_path=sample.wsi_path,
            feature_path=sample.feature_path,
        )
        if sample.question_type == "binary_yes_no":
            mode = "close-tf"
            question = sample.question.strip()
        else:
            mode = "close-mc"
            question = sample.format_closed_prompt()

        return self.generate_answer(
            question=question,
            mode=mode,
            slide_payload=slide_payload,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )

    @torch.inference_mode()
    def predict_caption(self, sample: CaptionSample, max_new_tokens: int = 512) -> str:
        """Run ALPaCA open-ended pathology report / caption generation (`open` mode)."""
        slide_payload = self._get_slide_features(
            slide_id=sample.slide_id,
            cohort=sample.cohort,
            wsi_path=sample.wsi_path,
            feature_path=sample.feature_path,
        )
        return self.generate_answer(
            question=sample.format_caption_prompt(),
            mode="open",
            slide_payload=slide_payload,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    def metadata(self) -> Dict[str, Any]:
        """Return ALPaCA model configuration metadata for reproducibility logging."""
        return {
            "model_name": self.model_name,
            "checkpoint_path": str(self.ckpt_path),
            "base_llm_path": str(self.paths.llama3_dir),
            "prototypes_dir": str(self.paths.alpaca_proto_dir),
            "agg_strategy": "longnet,gmm",
            "n_heads": [32, 16, 8],
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
