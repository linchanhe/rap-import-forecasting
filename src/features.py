from __future__ import annotations
import pandas as pd
from .config import CFG

def make_lag_features(df_pair_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create features:
    - dy = diff(target)
    - x kept in levels
    - lags = CFG.lags for both dy and x
    Returns:
      X_feat: DataFrame of lagged features
      y_target: Series of target_diff aligned with X_feat
    """
    target_col, x_col = CFG.target_col, CFG.x_col
    lags = CFG.lags

    df_trans = df_pair_level.copy()
    df_trans["dy"] = df_trans[target_col].diff()
    df_trans["x"] = df_trans[x_col]
    df_trans = df_trans.dropna()

    data = pd.DataFrame(index=df_trans.index)
    data["target_diff"] = df_trans["dy"]

    for l in range(1, lags + 1):
        data[f"dy_lag{l}"] = df_trans["dy"].shift(l)
        data[f"x_lag{l}"] = df_trans["x"].shift(l)

    data = data.dropna()
    X_feat = data.drop(columns=["target_diff"])
    y_target = data["target_diff"]

    return X_feat, y_target

def split_features_target(
    X_feat: pd.DataFrame,
    y_target: pd.Series,
    cutoff_date,
    test_start
) -> dict:
    """Split X/y into initial train and test (no peeking)."""
    X_train_init = X_feat.loc[:cutoff_date]
    y_train_init = y_target.loc[:cutoff_date]
    X_test = X_feat.loc[test_start:]
    y_test = y_target.loc[test_start:]

    return {
        "X_train_init": X_train_init,
        "y_train_init": y_train_init,
        "X_test": X_test,
        "y_test": y_test,
    }
