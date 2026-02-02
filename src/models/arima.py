from __future__ import annotations
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def select_arima_order(train_y_level: pd.Series, p_max=2, q_max=2) -> tuple[int, int, int]:
    """
    Grid search ARIMA(p,1,q) over p=0..p_max, q=0..q_max; choose by BIC.
    """
    ic_table = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            try:
                m = ARIMA(train_y_level, order=(p, 1, q)).fit()
                ic_table.append(((p, 1, q), float(m.bic)))
            except Exception:
                continue

    if not ic_table:
        raise RuntimeError("ARIMA grid search failed for all candidate orders.")

    best_order = sorted(ic_table, key=lambda x: x[1])[0][0]
    return best_order

def forecast_arima_recursive(train_y_level: pd.Series, test_y_level: pd.Series, order) -> pd.Series:
    """
    Recursive 1-step forecast in levels with re-fitting each step.
    """
    history = list(train_y_level.values)
    preds = []

    for t in range(len(test_y_level)):
        model = ARIMA(history, order=order).fit()
        preds.append(float(model.forecast()[0]))
        history.append(float(test_y_level.iloc[t]))

    return pd.Series(preds, index=test_y_level.index, name="arima_forecast")
