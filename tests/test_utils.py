import pandas as pd

from utils import ensure_outdir, plot_roc, save_csv


def test_ensure_outdir_creates_nested_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_outdir(target)
    assert result == target
    assert target.is_dir()


def test_save_csv_creates_parents_and_writes_data(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    out_path = tmp_path / "nested" / "out.csv"

    result = save_csv(df, out_path)

    assert result == out_path
    assert out_path.is_file()
    loaded = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(loaded, df)


def test_plot_roc_writes_png(tmp_path):
    out_path = tmp_path / "roc.png"
    plot_roc([0.0, 0.5, 1.0], [0.0, 0.8, 1.0], out_path)
    assert out_path.is_file()
    assert out_path.stat().st_size > 0
