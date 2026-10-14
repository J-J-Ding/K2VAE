"""Plot an ETT hourly forecast from a trained K2VAE checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_FILENAMES = {
    "etth1": "ETTh1.csv",
    "etth2": "ETTh2.csv",
}


def dataset_csv_path(data_root: Path, dataset_name: str) -> Path:
    """Return the local CSV path for a supported ETT hourly dataset."""
    return data_root / "ETT-small" / DATASET_FILENAMES[dataset_name.lower()]


def ensure_project_root_on_path() -> None:
    """Make project imports work when this file is executed by path."""
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def build_plot_frame(
    history: np.ndarray,
    actual: np.ndarray,
    samples: np.ndarray,
    variable_index: int,
) -> pd.DataFrame:
    """Combine one forecast window into a plotting-friendly table."""
    history_values = history[:, variable_index]
    actual_values = actual[:, variable_index]
    sample_values = samples[:, :, variable_index]
    forecast_mean = sample_values.mean(axis=0)
    lower = np.quantile(sample_values, 0.1, axis=0)
    upper = np.quantile(sample_values, 0.9, axis=0)

    history_length = len(history_values)
    prediction_length = len(actual_values)
    history_steps = np.arange(-history_length + 1, 1)
    future_steps = np.arange(1, prediction_length + 1)

    return pd.DataFrame(
        {
            "step": np.concatenate([history_steps, future_steps]),
            "history": np.concatenate(
                [history_values, np.full(prediction_length, np.nan)]
            ),
            "actual": np.concatenate(
                [np.full(history_length, np.nan), actual_values]
            ),
            "forecast_mean": np.concatenate(
                [np.full(history_length, np.nan), forecast_mean]
            ),
            "lower_10": np.concatenate([np.full(history_length, np.nan), lower]),
            "upper_90": np.concatenate([np.full(history_length, np.nan), upper]),
        }
    )


def save_forecast_plot(
    frame: pd.DataFrame,
    png_path: Path,
    csv_path: Path,
    variable_name: str,
    dataset_name: str = "etth1",
) -> None:
    """Save a forecast line chart and its underlying values."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    history = frame.dropna(subset=["history"])
    future = frame.dropna(subset=["actual"])
    boundary_step = history["step"].iloc[-1]
    boundary_value = history["history"].iloc[-1]
    connected_steps = np.concatenate([[boundary_step], future["step"].to_numpy()])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        history["step"],
        history["history"],
        color="#374151",
        linewidth=1.8,
        label="History",
    )
    ax.plot(
        connected_steps,
        np.concatenate([[boundary_value], future["actual"].to_numpy()]),
        color="#2563eb",
        linewidth=2,
        label="Actual",
    )
    ax.plot(
        connected_steps,
        np.concatenate([[boundary_value], future["forecast_mean"].to_numpy()]),
        color="#ea580c",
        linewidth=2,
        label="Forecast mean",
    )
    ax.fill_between(
        future["step"],
        future["lower_10"],
        future["upper_90"],
        color="#fb923c",
        alpha=0.22,
        label="10%-90% interval",
    )
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title(f"K2VAE {dataset_name.upper()} Forecast - {variable_name}")
    ax.set_xlabel("Step (forecast starts after 0)")
    ax.set_ylabel(variable_name)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_prediction(
    checkpoint_path: Path,
    data_root: Path,
    dataset_name: str,
    variable_name: str,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run the first ETT test window and return denormalized arrays."""
    ensure_project_root_on_path()
    import torch
    from torch.utils.data import DataLoader

    from probts.data import ProbTSBatchData
    from probts.data.data_manager import DataManager
    from probts.model.forecast_module import ProbTSForecastModule

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_args = dict(
        checkpoint["datamodule_hyper_parameters"]["data_manager"]["init_args"]
    )
    data_args["path"] = str(data_root)
    data_args["dataset"] = dataset_name
    data_manager = DataManager(**data_args)

    columns = pd.read_csv(dataset_csv_path(data_root, dataset_name), nrows=0).columns.tolist()[1:]
    if variable_name not in columns:
        raise ValueError(
            f"Unknown variable {variable_name!r}; choose one of: {', '.join(columns)}"
        )
    variable_index = columns.index(variable_name)

    batch = next(
        iter(DataLoader(data_manager.test_iter_dataset, batch_size=1, num_workers=0))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    batch_data = ProbTSBatchData(batch, device)

    model = ProbTSForecastModule.load_from_checkpoint(
        checkpoint_path,
        scaler=data_manager.scaler,
        no_training=True,
    ).to(device)
    model.eval()

    context_length = data_manager.context_length
    history = batch_data.past_target_cdf[:, -context_length:, :].clone()
    actual = batch_data.future_target_cdf.clone()
    batch_data.past_target_cdf = data_manager.scaler.transform(
        batch_data.past_target_cdf
    )
    with torch.no_grad():
        normalized_samples = model.forecaster.forecast(batch_data, num_samples)
        samples = data_manager.scaler.inverse_transform(normalized_samples)

    return (
        history[0].cpu().numpy(),
        actual[0].cpu().numpy(),
        samples[0].cpu().numpy(),
        variable_index,
    )


def newest_checkpoint(results_root: Path) -> Path:
    checkpoints = list(results_root.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found below {results_root}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset", choices=sorted(DATASET_FILENAMES), default="etth1")
    parser.add_argument("--variable", default="OT")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or newest_checkpoint(Path("results/smoke"))
    history, actual, samples, variable_index = generate_prediction(
        checkpoint_path,
        args.data_root,
        args.dataset,
        args.variable,
        args.num_samples,
    )
    frame = build_plot_frame(history, actual, samples, variable_index)
    stem = f"{args.dataset}_{args.variable.lower()}_forecast"
    png_path = args.output_dir / f"{stem}.png"
    csv_path = args.output_dir / f"{stem}.csv"
    save_forecast_plot(
        frame, png_path, csv_path, args.variable, dataset_name=args.dataset
    )
    print(f"Saved chart: {png_path.resolve()}")
    print(f"Saved data:  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
