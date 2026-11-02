"""Abstract base interface for pathology Whole-Slide Image Vision-Language Models."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.datasets.slidebench_vqa import VQASample


class BaseWSIVLM(ABC):
    """Standardized inference interface for WSI Vision-Language Models."""

    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights, encoders, projector, and tokenizer onto target device."""

    @abstractmethod
    def predict_vqa(self, sample: VQASample, max_new_tokens: int = 128) -> str:
        """Generate a raw text response for a single closed-ended or binary VQA sample.

        Args:
            sample: Standardized VQASample containing slide features/path and prompt.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Decoded model response string.
        """

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return model configuration metadata for reproducibility logging."""
