from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Paths and deterministic settings for the public training workflow."""

    processed_file: Path
    artifacts_dir: Path
    random_seed: int = 42

    @classmethod
    def from_repo_root(cls, repo_root: Path | None = None) -> "TrainingConfig":
        root = repo_root or Path(__file__).resolve().parents[2]
        return cls(
            processed_file=root / "dataset" / "ttc_delays_processed.csv",
            artifacts_dir=root / "server" / "model_artifacts",
        )
