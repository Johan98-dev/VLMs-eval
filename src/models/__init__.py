"""Unified model interface and WSI VLM adapters (SlideChat, PRISM2, MLLM-HWSI, ALPaCA)."""

from src.models.alpaca import ALPaCAAdapter
from src.models.base import BaseWSIVLM

__all__ = ["ALPaCAAdapter", "BaseWSIVLM"]



