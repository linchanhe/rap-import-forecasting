from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def reconstruct_levels_from_diffs(
    df_pair_level: pd.DataFrame,
    target_col: str,
    dates: pd.Index,
    preds_diff: list[float],
) -> tuple[pd.Series, pd.Series]:
    """
    Reconstruct level forecasts from 1-step diff predictions:
    yhat_t(level) = y_{t-1}(level) + yhat_t(diff)
    Returns:
      pred_levels: Series indexed by dates
      actual_levels: Series indexed by dates
    """
    pred_levels = []
    actual_levels = []

    for i, dt in enumerate(dates):
        loc = df_pair_level.index.get_loc(dt)
        prev_level = df_pair_level[target_col].iloc[loc - 1]
        pred_lvl = prev_level + preds_diff[i]
        pred_levels.append(pred_lvl)
        actual_levels.append(df_pair_level.loc[dt, target_col])

    pred_s = pd.Series(pred_levels, index=dates, name="pred_level")
    act_s = pd.Series(actual_levels, index=dates, name="actual_level")
    return pred_s, act_s
