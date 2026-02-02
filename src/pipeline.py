from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import warnings

from .report import build_report_from_outputs

from .config import CFG
from .io import load_excel, build_master, select_pair, train_test_split_index
from .features import make_lag_features, split_features_target
from .eval import reconstruct_levels_from_diffs, rmse
from .plot import save_comparison_plot

from .models.nn import select_nn_params, forecast_nn_recursive
from .models.arima import select_arima_order, forecast_arima_recursive
from .models.var import select_var_lag, forecast_var_recursive

from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def ensure_outputs_dir() -> Path:
    CFG.outputs_dir.mkdir(parents=True, exist_ok=True)
    return CFG.outputs_dir

def run_all() -> dict:
    outdir = ensure_outputs_dir()

    # ---------- 0) Load data ----------
    df_y, df_x = load_excel(CFG.data_path, CFG.date_col)
    df_master = build_master(df_y, df_x)
    df_pair_level = select_pair(df_master)

    split = train_test_split_index(df_pair_level)
    cutoff_date = split["cutoff_date"]
    test_start = split["test_start"]
    test_end = split["test_end"]
    test_idx = split["test_idx"]

    # ---------- 1) Features for NN ----------
    X_feat, y_target = make_lag_features(df_pair_level)
    parts = split_features_target(X_feat, y_target, cutoff_date=cutoff_date, test_start=test_start)

    X_train_init = parts["X_train_init"]
    y_train_init = parts["y_train_init"]
    X_test = parts["X_test"]
    y_test = parts["y_test"]

    # ---------- 2) Neural Network ----------
    best_params = select_nn_params(X_train_init, y_train_init, random_state=CFG.random_state)
    preds_diff = forecast_nn_recursive(
        X_train_init, y_train_init, X_test, y_test, best_params, random_state=CFG.random_state
    )

    nn_dates = X_test.index
    nn_forecast_level, actual_levels = reconstruct_levels_from_diffs(
        df_pair_level, CFG.target_col, nn_dates, preds_diff
    )
    rmse_nn = rmse(actual_levels.values, nn_forecast_level.values)

    # ---------- 3) ARIMA benchmark ----------
    train_y_level = df_pair_level.loc[:cutoff_date, CFG.target_col]
    test_y_level = df_pair_level.loc[test_start:, CFG.target_col]

    best_order = select_arima_order(train_y_level, p_max=2, q_max=2)
    arima_forecast = forecast_arima_recursive(train_y_level, test_y_level, best_order)
    rmse_arima = rmse(test_y_level.values, arima_forecast.values)

    # ---------- 4) VAR benchmark ----------
    train_pair = df_pair_level.loc[:cutoff_date].dropna()
    selected_lag = select_var_lag(train_pair, maxlags=8)
    var_forecast = forecast_var_recursive(
        df_pair_level=df_pair_level,
        train_end=cutoff_date,
        test_idx=test_idx,
        selected_lag=selected_lag,
        target_col=CFG.target_col
    )
    actual_var = df_pair_level.loc[test_idx, CFG.target_col]
    rmse_var = rmse(actual_var.values, var_forecast.values)

    # ---------- 5) Save outputs ----------
    # RMSE CSV
    rmse_df = pd.DataFrame({
        "Model": ["ARIMA", "VAR", "Neural Network"],
        "RMSE": [rmse_arima, rmse_var, rmse_nn],
    })
    rmse_path = outdir / "rmse.csv"
    rmse_df.to_csv(rmse_path, index=False)

    # Forecasts CSV
    pd.DataFrame({
        "date": nn_dates,
        "actual_level": actual_levels.values,
        "nn_forecast": nn_forecast_level.values,
        "arima_forecast": arima_forecast.reindex(nn_dates).values,
        "var_forecast": var_forecast.reindex(nn_dates).values,
    }).to_csv(outdir / "forecasts.csv", index=False)

    # Plot
    plot_path = outdir / "forecast_comparison.png"
    save_comparison_plot(
        dates=nn_dates,
        actual_levels=actual_levels,
        arima_forecast=arima_forecast.reindex(nn_dates),
        var_forecast=var_forecast.reindex(nn_dates),
        nn_forecast=nn_forecast_level,
        rmse_arima=rmse_arima,
        rmse_var=rmse_var,
        rmse_nn=rmse_nn,
        outpath=plot_path
    )

    # ---------- 6) Write Metadata & Generate Report ----------
    # --- Report generation ---
    # Option A: if you already generated html_path/pdf_path earlier, keep them
    # Option B: if you generate report from outputs, capture returned paths

    report_html_path = outdir / "report.html"
    report_pdf_path = outdir / "report.pdf"

    # If your code calls something like build_report_from_outputs(outdir),
    # make sure it returns the actual paths. Example:
    try:
        print("Generating report from outputs...")
        from .report import build_report_from_outputs
        html_p, pdf_p = build_report_from_outputs(outdir)
        report_html_path = html_p
        report_pdf_path = pdf_p if pdf_p is not None else report_pdf_path
    except Exception:
        # fallback: assume pipeline already wrote report.html / report.pdf
        pass

    # --- Write metadata.json and return meta ---
    meta_out = {
        "data_path": str(CFG.data_path),
        "n_total": split["n_total"],
        "n_train": split["n_train"],
        "cutoff_date": str(cutoff_date.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "nn_best_params": best_params,
        "arima_best_order": list(best_order),
        "var_selected_lag": selected_lag,
        "outputs": {
            "rmse_csv": str(rmse_path),
            "plot_png": str(plot_path),
            "report_html": str(report_html_path),
            "report_pdf": str(report_pdf_path) if report_pdf_path.exists() else None,
        }
    }
    
    (outdir / "metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    # ---- Standard output summary ----
    print("Outputs written to: outputs/")
    print("- rmse.csv")
    print("- forecast_comparison.png")
    print("- report.html")
    if meta_out["outputs"]["report_pdf"] is not None:
        print("- report.pdf")
    else:
        print("- report.pdf (not generated)")


    return meta_out

if __name__ == "__main__":
    meta = run_all()
    print("Pipeline finished.")