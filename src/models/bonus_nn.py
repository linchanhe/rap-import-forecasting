from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def bonus_trim_fill_dataset(
    df_master: pd.DataFrame,
    df_y: pd.DataFrame,
    date_col: str,
    target_col: str,
    analysis_end: pd.Timestamp,
    candidate_cols: list[str],
) -> pd.DataFrame:
    cols_to_use = [c for c in candidate_cols if c in df_master.columns]
    df_b = pd.merge(df_y[[date_col, target_col]], df_master[cols_to_use], on=date_col, how="left")
    df_b = df_b.set_index(date_col).sort_index()
    df_b = df_b[df_b.index <= analysis_end]
    df_b = df_b.fillna(method="ffill", limit=2).dropna()
    return df_b

def bonus_features(df_b: pd.DataFrame, target_col: str, cols_to_use: list[str], lags: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    df_b = df_b.copy()
    df_b["dy"] = df_b[target_col].diff()

    data_b = pd.DataFrame(index=df_b.index)
    data_b["target_diff"] = df_b["dy"]

    for l in range(1, lags + 1):
        data_b[f"dy_lag{l}"] = df_b["dy"].shift(l)
        for c in cols_to_use:
            if df_b[c].mean() > 50:
                feat = df_b[c].diff().shift(l)
            else:
                feat = df_b[c].shift(l)
            data_b[f"{c}_lag{l}"] = feat

    data_b = data_b.dropna()
    X_b = data_b.drop(columns=["target_diff"])
    y_b = data_b["target_diff"]
    return X_b, y_b

def bonus_forecast_nn_recursive(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_all: pd.Series,
    b_params: dict,
    random_state: int,
) -> list[float]:
    preds = []
    history_X = X_tr.copy()
    history_y = y_tr.copy()

    for date in X_te.index:
        scaler = StandardScaler()
        h_X_scaled = scaler.fit_transform(history_X)

        model = MLPRegressor(**b_params, max_iter=3000, random_state=random_state)
        model.fit(h_X_scaled, history_y)

        curr = X_te.loc[[date]]
        curr_sc = scaler.transform(curr)
        pred = float(model.predict(curr_sc)[0])
        preds.append(pred)

        if date in y_all.index:
            history_X = pd.concat([history_X, curr])
            history_y = pd.concat([history_y, y_all.loc[[date]]])

    return preds
