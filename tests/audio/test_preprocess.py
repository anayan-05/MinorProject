"""Unit Tests for Audio Preprocessing Module.
Owner: Debajyoti (Audio ML Lead)
"""

import pytest
import numpy as np

from ml.audio.src.preprocess import AudioPreprocessor

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None


@pytest.fixture
def preprocessor():
    return AudioPreprocessor(
        sample_rate=16000,
        target_duration=3.0,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
    )


def test_preprocessor_initialization(preprocessor):
    assert preprocessor.sample_rate == 16000
    assert preprocessor.target_duration == 3.0
    assert preprocessor.target_length == 48000
    assert preprocessor.n_mels in [64, 128]


def test_standardise_stereo_to_mono(preprocessor):
    # Stereo input (2 channels, 50,000 samples at 16kHz)
    if HAS_TORCH:
        stereo_input = torch.randn(2, 50000)
    else:
        stereo_input = np.random.randn(2, 50000).astype(np.float32)

    standardised = preprocessor.load_and_standardise(stereo_input, input_sample_rate=16000)

    assert standardised.ndim == 2
    assert standardised.shape[0] == 1  # Converted to mono
    assert standardised.shape[1] == 48000  # Clamped to 3.0s (48,000 samples)


def test_standardise_padding_short_audio(preprocessor):
    # Short audio (1 channel, 16,000 samples = 1.0s)
    if HAS_TORCH:
        short_input = torch.randn(1, 16000)
    else:
        short_input = np.random.randn(1, 16000).astype(np.float32)

    standardised = preprocessor.load_and_standardise(short_input, input_sample_rate=16000)

    assert standardised.shape[0] == 1
    assert standardised.shape[1] == 48000  # Zero-padded to 48,000 samples

    if HAS_TORCH and isinstance(standardised, torch.Tensor):
        assert torch.all(standardised[:, 16000:] == 0.0)
    else:
        assert np.all(standardised[:, 16000:] == 0.0)


def test_compute_log_mel_spectrogram_shape_and_normalisation(preprocessor):
    if HAS_TORCH:
        dummy_waveform = torch.randn(1, 48000)
    else:
        dummy_waveform = np.random.randn(1, 48000).astype(np.float32)

    log_mel = preprocessor.compute_log_mel(dummy_waveform)

    # Shape should be [1, n_mels, time_frames=94]
    assert log_mel.ndim >= 2
    assert log_mel.shape[-2] == preprocessor.n_mels
    assert log_mel.shape[-1] >= 90  # 94 time frames for 3.0s window with hop 512

    # Standardisation check: mean near 0, std near 1
    if HAS_TORCH and isinstance(log_mel, torch.Tensor):
        assert abs(log_mel.mean().item()) < 0.1
        assert abs(log_mel.std().item() - 1.0) < 0.1
    else:
        assert abs(np.mean(log_mel)) < 0.1
        assert abs(np.std(log_mel) - 1.0) < 0.1


def test_add_background_noise_snr(preprocessor):
    clean_signal = preprocessor.generate_synthetic_cue("screaming_shouting")
    noisy_signal = preprocessor.add_background_noise(clean_signal, snr_db=15.0)

    assert noisy_signal.shape == clean_signal.shape
    if HAS_TORCH and isinstance(clean_signal, torch.Tensor):
        assert not torch.allclose(clean_signal, noisy_signal)
        assert torch.max(torch.abs(noisy_signal)) <= 1.0
    else:
        assert not np.allclose(clean_signal, noisy_signal)
        assert np.max(np.abs(noisy_signal)) <= 1.0


def test_synthetic_cue_generation_all_classes(preprocessor):
    classes = [
        "background_normal",
        "screaming_shouting",
        "alarm_siren",
        "glass_breaking",
        "impact_crash",
    ]
    for c in classes:
        wf = preprocessor.generate_synthetic_cue(c)
        assert wf.ndim == 2
        assert wf.shape[0] == 1
        assert wf.shape[1] == 48000
        if HAS_TORCH and isinstance(wf, torch.Tensor):
            assert torch.max(torch.abs(wf)) > 0.0
        else:
            assert np.max(np.abs(wf)) > 0.0

    with pytest.raises(ValueError):
        preprocessor.generate_synthetic_cue("invalid_unsupported_cue")
