from src.io import load_excel, build_master, select_pair
from src.features import make_lag_features
from src.config import CFG

def test_make_lag_features_shape_and_na():
    df_y, df_x = load_excel(CFG.data_path, CFG.date_col)
    master = build_master(df_y, df_x)
    pair = select_pair(master)

    X, y = make_lag_features(pair)
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0

    # Expect 2*lags columns: dy_lag1..L and x_lag1..L
    assert X.shape[1] == 2 * CFG.lags
