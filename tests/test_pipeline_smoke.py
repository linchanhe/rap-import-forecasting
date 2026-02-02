from src.pipeline import run_all

def test_pipeline_runs_smoke():
    meta = run_all()
    assert "outputs" in meta
    assert meta["outputs"]["rmse_csv"] is not None
    assert meta["outputs"]["plot_png"] is not None
    assert meta["outputs"]["report_html"] is not None
