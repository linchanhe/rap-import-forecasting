from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_path: Path = project_root / "data" / "data.xlsx"
    outputs_dir: Path = project_root / "outputs"

    # Columns
    date_col: str = "date"
    target_col: str = "import_clv_qna_sa"
    x_col: str = "import_s_clv_qna_sa"

    # Time & split
    analysis_end: pd.Timestamp = pd.Timestamp("2025-04-01")
    train_ratio: float = 0.8

    # Features
    lags: int = 4

    # Reproducibility
    random_state: int = 42

CFG = Config()
