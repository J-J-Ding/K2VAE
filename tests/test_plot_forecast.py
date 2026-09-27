from pathlib import Path

import numpy as np

from scripts.plot_etth1_forecast import (
    build_plot_frame,
    ensure_project_root_on_path,
    save_forecast_plot,
)


def test_build_plot_frame_contains_history_actual_mean_and_interval():
    history = np.array([[1.0, 10.0], [2.0, 20.0]])
    actual = np.array([[3.0, 30.0], [4.0, 40.0]])
    samples = np.array(
        [
            [[2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0]],
        ]
    )

    frame = build_plot_frame(history, actual, samples, variable_index=1)

    assert frame["step"].tolist() == [-1, 0, 1, 2]
    assert frame["history"].iloc[:2].tolist() == [10.0, 20.0]
    assert frame["actual"].iloc[2:].tolist() == [30.0, 40.0]
    assert frame["forecast_mean"].iloc[2:].tolist() == [30.0, 40.0]
    assert np.allclose(frame["lower_10"].iloc[2:], [22.0, 32.0])
    assert np.allclose(frame["upper_90"].iloc[2:], [38.0, 48.0])


def test_save_forecast_plot_writes_png_and_csv(tmp_path: Path):
    history = np.array([[1.0], [2.0]])
    actual = np.array([[3.0], [4.0]])
    samples = np.array([[[2.5], [3.5]], [[3.5], [4.5]]])
    frame = build_plot_frame(history, actual, samples, variable_index=0)
    png_path = tmp_path / "forecast.png"
    csv_path = tmp_path / "forecast.csv"

    save_forecast_plot(frame, png_path, csv_path, variable_name="OT")

    assert png_path.stat().st_size > 0
    assert csv_path.read_text(encoding="utf-8").startswith("step,history,actual")


def test_ensure_project_root_on_path_restores_script_import_path(monkeypatch):
    import sys

    project_root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != project_root])

    ensure_project_root_on_path()

    assert sys.path[0] == project_root
