from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def save_comparison_plot(
    dates: pd.Index,
    actual_levels: pd.Series,
    arima_forecast: pd.Series,
    var_forecast: pd.Series,
    nn_forecast: pd.Series,
    rmse_arima: float,
    rmse_var: float,
    rmse_nn: float,
    outpath
) -> None:
    plt.figure(figsize=(14, 7))

    plt.plot(dates, actual_levels.values, label="Actual Data", linewidth=2, marker="o")
    plt.plot(dates, arima_forecast.values, label=f"ARIMA (RMSE {rmse_arima:.2f})", linestyle="--", marker="x", alpha=0.8)
    plt.plot(dates, var_forecast.values, label=f"VAR (RMSE {rmse_var:.2f})", linestyle="-.", marker="s", alpha=0.8)
    plt.plot(dates, nn_forecast.values, label=f"Neural Network (RMSE {rmse_nn:.2f})", linewidth=2.5, marker="d")

    plt.title("Forecast Comparison", fontsize=14)
    plt.ylabel("Imports (Level)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
