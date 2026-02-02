from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def select_nn_params(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> dict:
    """
    Grid search on training set with time-series CV.
    Returns best_params dict.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        "hidden_layer_sizes": [(5,), (10,)],
        "activation": ["relu", "tanh"],
        "alpha": [0.01],
        "learning_rate_init": [0.005, 0.01],
    }

    mlp = MLPRegressor(max_iter=3000, random_state=random_state)
    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        mlp,
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    return dict(grid.best_params_)

def forecast_nn_recursive(
    X_train_init: pd.DataFrame,
    y_train_init: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: dict,
    random_state: int
) -> list[float]:
    """
    Recursive 1-step ahead forecasts for diffs with re-estimation each step.
    Standardize each step using only history.
    Returns list of predicted diffs.
    """
    preds_diff = []
    history_X = X_train_init.copy()
    history_y = y_train_init.copy()

    for i in range(len(X_test)):
        scaler_rec = StandardScaler()
        hist_X_scaled = scaler_rec.fit_transform(history_X)

        model = MLPRegressor(**best_params, max_iter=5000, random_state=random_state)
        model.fit(hist_X_scaled, history_y)

        curr_X = X_test.iloc[[i]]
        curr_X_scaled = scaler_rec.transform(curr_X)
        pred = float(model.predict(curr_X_scaled)[0])
        preds_diff.append(pred)

        # update with realized diff (no peeking beyond current)
        history_X = pd.concat([history_X, curr_X])
        history_y = pd.concat([history_y, y_test.iloc[[i]]])

    return preds_diff
