"""Unit Tests for Compact Audio CNN Model Architecture.
Owner: Debajyoti (Audio ML Lead)
"""

import os
import pytest
import numpy as np

from ml.audio.src.model import PathPilotAudioCNN

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None


@pytest.fixture
def model():
    return PathPilotAudioCNN(num_classes=5)


def test_model_forward_single_and_batched(model):
    model.eval()

    if HAS_TORCH:
        x_single = torch.randn(1, 64, 94)
        out_single = model(x_single)
        assert out_single.shape == (1, 5)

        x_batch = torch.randn(4, 1, 64, 94)
        out_batch = model(x_batch)
        assert out_batch.shape == (4, 5)
    else:
        x_single = np.random.randn(1, 64, 94).astype(np.float32)
        out_single = model(x_single)
        assert out_single.shape == (1, 5)

        x_batch = np.random.randn(4, 1, 64, 94).astype(np.float32)
        out_batch = model(x_batch)
        assert out_batch.shape == (4, 5)


def test_model_predict_proba_properties(model):
    if HAS_TORCH:
        x = torch.randn(2, 1, 64, 94)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 5)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    else:
        x = np.random.randn(2, 1, 64, 94).astype(np.float32)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 5)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        sums = np.sum(probs, axis=-1)
        assert np.allclose(sums, np.ones_like(sums), atol=1e-4)


def test_model_footprint_and_parameter_budget(model):
    param_counts = model.count_parameters()
    total_params = param_counts["total_parameters"]
    size_mb = model.get_model_size_mb()

    # Parameter budget constraint: Must be compact for mobile CPU (< 200k params)
    assert total_params < 200000, f"Model has {total_params} parameters, exceeding budget"
    assert total_params > 50000, f"Model has {total_params} parameters, unexpectedly small"

    # Memory footprint constraint: Must be well below 5 MB
    assert size_mb < 5.0, f"Model size is {size_mb} MB, exceeding 5 MB constraint"
    assert size_mb < 1.0, f"Model size is {size_mb} MB, should be under 1 MB"


def test_model_export_torchscript_and_onnx(model, tmp_path):
    ts_path = os.path.join(tmp_path, "test_model.torchscript.pt")
    onnx_path = os.path.join(tmp_path, "test_model.onnx")

    exported_ts = model.export_torchscript(ts_path)
    assert os.path.exists(exported_ts)
    assert os.path.getsize(exported_ts) > 100

    exported_onnx = model.export_onnx(onnx_path)
    assert os.path.exists(exported_onnx)
    assert os.path.getsize(exported_onnx) > 100
