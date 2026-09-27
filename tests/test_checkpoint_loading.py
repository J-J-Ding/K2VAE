import torch

from probts.data.data_utils.data_scaler import StandardScaler
from probts.model.forecaster.forecaster import Forecaster
from probts.utils.save_utils import load_checkpoint


class DummyModel:
    def __init__(self, forecaster, scaler=None, **kwargs):
        self.forecaster = forecaster
        self.scaler = scaler

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


def test_loads_trusted_lightning_checkpoint_with_project_objects(tmp_path):
    forecaster = Forecaster(
        target_dim=1,
        context_length=2,
        prediction_length=1,
        freq="h",
        lags_list=[],
    )
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "forecaster": forecaster,
                "scaler": StandardScaler(),
            },
            "state_dict": {},
        },
        checkpoint_path,
    )

    model = load_checkpoint(DummyModel, checkpoint_path)

    assert isinstance(model.forecaster, Forecaster)
    assert model.loaded_state_dict == {}
