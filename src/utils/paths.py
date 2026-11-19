"""Cross-platform external path resolution for datasets, models, and evaluation results."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT: Path = PROJECT_ROOT.parent


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved directory paths for datasets, model weights, and evaluation outputs."""

    project_root: Path = PROJECT_ROOT
    datasets_dir: Path = EXTERNAL_ROOT / "Datasets" / "SlideBench"
    models_dir: Path = EXTERNAL_ROOT / "Models"
    results_dir: Path = PROJECT_ROOT / "results"

    @property
    def slidechat_dir(self) -> Path:
        """Return path to SlideChat checkpoint directory."""
        return self.models_dir / "SlideChat"

    @property
    def qwen_dir(self) -> Path:
        """Return path to Qwen2.5-7B-Instruct base LLM directory."""
        return self.models_dir / "Qwen2.5-7B-Instruct"

    @property
    def prism2_dir(self) -> Path:
        """Return path to PRISM2 checkpoint directory."""
        return self.models_dir / "PRISM2"

    @property
    def mllm_hwsi_dir(self) -> Path:
        """Return path to MLLM-HWSI checkpoint directory."""
        return self.models_dir / "MLLM-HWSI"

    @property
    def alpaca_dir(self) -> Path:
        """Return path to ALPaCA (Llama-slideQA) checkpoint directory."""
        return self.models_dir / "ALPaCA"

    @property
    def alpaca_repo_dir(self) -> Path:
        """Return path to cloned upstream ZeyuGaoAi/ALPaCA code and config directory."""
        return self.alpaca_dir / "repo"

    @property
    def alpaca_proto_dir(self) -> Path:
        """Return path to ALPaCA GTEx-TCGA GMM prototypes (more_version: 256/128/64)."""
        return (
            self.alpaca_repo_dir
            / "GMM_feature_extraction"
            / "GTEx-TCGA-prototypes"
            / "GTEx_TCGA_merged_output_more"
        )

    @property
    def alpaca_sample_features_dir(self) -> Path:
        """Return path to CNX-PathLLM/Llama-slideQA-Sample-Features (40-slide reference set)."""
        return self.alpaca_dir / "sample_features"

    @property
    def llama3_dir(self) -> Path:
        """Return path to meta-llama/Llama-3.1-8B-Instruct base LLM directory."""
        return self.models_dir / "Llama-3.1-8B-Instruct"

    @property
    def encoders_dir(self) -> Path:
        """Return path to foundation encoders root directory."""
        return self.models_dir / "Encoders"

    @property
    def virchow2_dir(self) -> Path:
        """Return path to Virchow2 encoder directory."""
        return self.encoders_dir / "Virchow2"

    @property
    def conch_dir(self) -> Path:
        """Return path to CONCH encoder directory."""
        return self.encoders_dir / "CONCH"

    @property
    def hipt_dir(self) -> Path:
        """Return path to HIPT encoders directory."""
        return self.encoders_dir / "HIPT"

    @property
    def cellvit_dir(self) -> Path:
        """Return path to CellViT encoder directory."""
        return self.encoders_dir / "CellViT"

