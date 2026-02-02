from src.io import load_excel, build_master, select_pair
from src.config import CFG

def test_load_and_select_pair_smoke():
    df_y, df_x = load_excel(CFG.data_path, CFG.date_col)
    assert CFG.date_col in df_y.columns
    assert CFG.date_col in df_x.columns

    master = build_master(df_y, df_x)
    assert master.index.name == CFG.date_col

    pair = select_pair(master)
    assert CFG.target_col in pair.columns
    assert CFG.x_col in pair.columns
    assert len(pair) > 10  # sanity
