from __future__ import annotations
import pandas as pd
from statsmodels.tsa.api import VAR

def select_var_lag(train_pair_level: pd.DataFrame, maxlags=8) -> int:
    """
    Select VAR lag using BIC on differenced training data.
    """
    train_diff = train_pair_level.diff().dropna()
    model_var = VAR(train_diff)
    lag_selection = model_var.select_order(maxlags=maxlags)
    selected_lag = lag_selection.selected_orders["bic"]
    if selected_lag is None:
        # fallback
        selected_lag = 1
    return int(selected_lag)

def forecast_var_recursive(
    df_pair_level: pd.DataFrame,
    train_end,
    test_idx: pd.Index,
    selected_lag: int,
    target_col: str
) -> pd.Series:
    """
    Recursive 1-step forecast in levels via VAR on diffs, then reconstruct target level.
    """
    preds = []

    for dt in test_idx:
        prev_dt_idx = df_pair_level.index.get_loc(dt) - 1
        prev_dt = df_pair_level.index[prev_dt_idx]

        pair_upto = df_pair_level.loc[:prev_dt].dropna()
        diff_upto = pair_upto.diff().dropna()

        model = VAR(diff_upto)
        res = model.fit(selected_lag)

        fc_diff = res.forecast(diff_upto.values[-selected_lag:], steps=1)[0]

        target_pos = list(diff_upto.columns).index(target_col)
        prev_val = float(pair_upto.loc[prev_dt, target_col])
        yhat = prev_val + float(fc_diff[target_pos])

        preds.append(yhat)

    return pd.Series(preds, index=test_idx, name="var_forecast")
