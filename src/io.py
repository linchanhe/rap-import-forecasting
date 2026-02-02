from __future__ import annotations
import pandas as pd
from .config import CFG

def load_excel(path=CFG.data_path, date_col=CFG.date_col) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data from Excel sheets `data_y` and `data_x`."""
    df_y = pd.read_excel(path, sheet_name="data_y")
    df_x = pd.read_excel(path, sheet_name="data_x")

    df_y[date_col] = pd.to_datetime(df_y[date_col])
    df_x[date_col] = pd.to_datetime(df_x[date_col])
    return df_y, df_x

def build_master(df_y: pd.DataFrame, df_x: pd.DataFrame) -> pd.DataFrame:
    """Merge target sheet and predictors sheet on date; return indexed master frame."""
    date_col = CFG.date_col
    target_col = CFG.target_col

    df_y = df_y[df_y[date_col] <= CFG.analysis_end].copy()

    df_master = pd.merge(
        df_y[[date_col, target_col]],
        df_x,
        on=date_col,
        how="left",
    )

    df_master = df_master.set_index(date_col).sort_index()
    return df_master

def select_pair(df_master: pd.DataFrame) -> pd.DataFrame:
    """Extract main pair (target, x) in levels; drop NA."""
    pair = df_master[[CFG.target_col, CFG.x_col]].dropna().copy()
    pair = pair.sort_index()
    return pair

def train_test_split_index(df_pair_level: pd.DataFrame) -> dict:
    """Compute train/test split indices using CFG.train_ratio (chronological split)."""
    n_total = len(df_pair_level)
    n_train = int(n_total * CFG.train_ratio)

    train_idx = df_pair_level.index[:n_train]
    test_idx = df_pair_level.index[n_train:]

    cutoff_date = train_idx[-1]
    test_start = test_idx[0]
    test_end = test_idx[-1]

    return {
        "n_total": n_total,
        "n_train": n_train,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "cutoff_date": cutoff_date,
        "test_start": test_start,
        "test_end": test_end,
    }
