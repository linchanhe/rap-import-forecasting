# rap-import-forecasting
A fully reproducible analytical pipeline for forecasting Eurozone imports using Python and Nix.

![python](https://img.shields.io/badge/python-3.12-blue)
![env](https://img.shields.io/badge/environment-nix--shell-orange)
![run](https://img.shields.io/badge/entrypoint-./run-brightgreen)
![tests](https://img.shields.io/badge/tests-pytest-informational)

A **reproducible analytical pipeline** for **1-step-ahead forecasting** of the target series
`import_clv_qna_sa` (macroeconomic indicator), benchmarking:

- **ARIMA** (univariate baseline)
- **VAR** (multivariate baseline)
- **Neural Network (MLPRegressor)** with lagged features + time-series CV

The repository is designed so an instructor can **run everything with one command**, with minimal setup.

---

##  Quick start

You only need **Nix** installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/linchanhe/rap-import-forecasting.git
   cd rap-import-forecasting
2. From the repository root:

```bash
./run
````

This will:

1. enter the Nix environment,
2. run tests,
3. run the pipeline and generate outputs under `outputs/`.

You can also run only one stage:

```bash
./run test
./run pipeline
```

> Note: the first run may download Nix dependencies depending on your machine and network.

---

##  Expected outputs

After the pipeline finishes, you will get:

* `outputs/rmse.csv` — RMSE scoreboard (ARIMA / VAR / NN)
* `outputs/forecast_comparison.png` — forecast comparison plot
* `outputs/report.html` — HTML report

---

##  Project structure

```text
rap-import-forecasting/
├── default.nix            # pinned environment definition (Nix)
├── run                    # one-command entrypoint
├── data/
│   └── data.xlsx          # input dataset
├── src/
│   ├── config.py          # central configuration (paths, columns, split)
│   ├── pipeline.py        # pipeline orchestration
│   ├── io.py              # data ingestion + merging
│   ├── features.py        # transformations + lag features
│   ├── eval.py            # RMSE evaluation
│   ├── plot.py            # plotting
│   ├── report.py          # HTML/PDF report generator
│   └── models/
│       ├── nn.py          # neural network model (MLP)
│       ├── arima.py       # ARIMA baseline
│       ├── var.py         # VAR baseline
│       └── bonus_nn.py    # optional bonus model
└── tests/                 # pytest smoke tests
```

---

##  Methodology

### 1) Data processing
* Source: data/data.xlsx (~222KB).
* Target ($Y$): import_clv_qna_sa (Eurozone Real Imports). 
* The pipeline loads the relevant sheets/columns, parses the date column, and aligns a consistent sample period.
* Train/test split uses a fixed ratio (see `src/config.py`).

### 2) Feature engineering

* The target series is transformed to a stationary representation (first differences).
* Lagged features are created for both the target (diff) and predictor series.
* Forecasts are reconstructed back to **levels** for evaluation.

### 3) Forecasting setup

All models are evaluated using **recursive 1-step-ahead forecasting** on the test period:
refit → predict 1 step → append true observation → repeat.

### 4) Evaluation + reporting

* Primary metric: **RMSE in levels** over the test period
* The pipeline exports a plot, a RMSE table, and an HTML report (and optionally PDF)

---

##  Tests

Inside the Nix shell, you can run:

```bash
python -m pytest
```

Tests are lightweight smoke tests that ensure:

* data loading works,
* feature generation works,
* pipeline runs end-to-end and produces outputs.

---

##  Manual execution (without `./run`)

```bash
nix-shell
python -m pytest
python -m src.pipeline
```

---

## References

Course website (RAP4MADS): https://rap4mads.eu/  
This repository applies the reproducibility practices introduced there to a forecasting pipeline.

---
