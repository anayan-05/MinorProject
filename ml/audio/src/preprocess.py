"""Audio Preprocessing Module for PathPilot Audio ML Subsystem.
Owner: Debajyoti (Audio ML)

Dual-engine architecture:
- High-performance PyTorch/Torchaudio mode for training, Colab, and server environments.
- High-portability Pure-NumPy fallback mode for edge, mobile, or Windows environments
  where C-extension DLL loading is restricted by Application Control policies.

Handles:
- Sample rate standardisation (16 kHz)
- Mono channel conversion
- Fixed-duration windowing / padding (default: 3.0s -> 48,000 samples)
- Mel filterbank and Log-Mel Spectrogram extraction (64 Mel bins)
- Normalisation (mean / standard deviation or min-max)
- Data augmentations for noise robustness (Additive Gaussian Noise, SpecAugment)
- Synthetic audio waveform synthesis for testing without downloading external gigabytes
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union

# Attempt PyTorch import with graceful fallback
try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None
    torchaudio = None
    T = None


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 64,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Constructs a triangular Mel filterbank matrix of shape [n_mels, n_fft // 2 + 1]."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    num_bins = n_fft // 2 + 1
    filters = np.zeros((n_mels, num_bins), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        if center > left:
            for b in range(left, min(center, num_bins)):
                filters[i - 1, b] = (b - left) / (center - left)
        if right > center:
            for b in range(center, min(right, num_bins)):
                filters[i - 1, b] = (right - b) / (right - center)

    # Area normalisation
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filters = filters * enorm[:, np.newaxis]
    return filters


class AudioPreprocessor:
    """Standardised audio preprocessor for PathPilot acoustic cue models."""

    def __init__(
        self,
        sample_rate: int = 16000,
        target_duration: float = 3.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_length = int(sample_rate * target_duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        # NumPy filterbank matrix
        self.mel_filters = create_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        # PyTorch Torchaudio transform (if available)
        if HAS_TORCH:
            try:
                self.mel_transform = T.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    win_length=self.n_fft,
                    hop_length=self.hop_length,
                    f_min=self.f_min,
                    f_max=self.f_max,
                    n_mels=self.n_mels,
                    power=2.0,
                )
                self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)
            except Exception:
                self.mel_transform = None
                self.amplitude_to_db = None
        else:
            self.mel_transform = None
            self.amplitude_to_db = None

    def load_and_standardise(
        self,
        audio_input: Any,
        input_sample_rate: Optional[int] = None,
    ) -> Any:
        """Loads audio from file path or tensor/array, converts to mono 16kHz,

        and standardises length to target_length (48,000 samples for 3.0s).
        Returns array/tensor of shape [1, target_length].
        """
        # Convert input to numpy array
        if HAS_TORCH and isinstance(audio_input, torch.Tensor):
            arr = audio_input.detach().cpu().numpy()
            return_tensor = True
        elif isinstance(audio_input, np.ndarray):
            arr = audio_input.copy()
            return_tensor = HAS_TORCH
        elif isinstance(audio_input, str):
            # Read WAV using standard library wave module or torchaudio
            import wave
            with wave.open(audio_input, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            raw_arr = np.frombuffer(raw_data, dtype=dtype_map.get(sampwidth, np.int16)).astype(np.float32)
            if n_channels > 1:
                raw_arr = raw_arr.reshape(-1, n_channels).T
            else:
                raw_arr = raw_arr.reshape(1, -1)
            raw_arr = raw_arr / (2.0 ** (8 * sampwidth - 1))
            arr = raw_arr
            input_sample_rate = sr
            return_tensor = HAS_TORCH
        else:
            arr = np.array(audio_input, dtype=np.float32)
            return_tensor = False

        sr = input_sample_rate or self.sample_rate

        # Ensure 2D shape: [channels, time]
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim > 2:
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]

        # 1. Convert multi-channel to mono
        if arr.shape[0] > 1:
            arr = np.mean(arr, axis=0, keepdims=True)

        # 2. Resample if necessary (using linear interpolation)
        if sr != self.sample_rate and arr.shape[1] > 0:
            target_samples = int(arr.shape[1] * (self.sample_rate / sr))
            orig_indices = np.linspace(0, arr.shape[1] - 1, target_samples)
            arr = np.interp(orig_indices, np.arange(arr.shape[1]), arr[0])[np.newaxis, :]

        # 3. Standardise length to target_length
        curr_len = arr.shape[1]
        if curr_len > self.target_length:
            arr = arr[:, : self.target_length]
        elif curr_len < self.target_length:
            pad_amount = self.target_length - curr_len
            arr = np.pad(arr, ((0, 0), (0, pad_amount)), mode="constant", constant_values=0.0)

        arr = arr.astype(np.float32)

        if HAS_TORCH and return_tensor:
            return torch.from_numpy(arr)
        return arr

    def compute_log_mel_numpy(self, waveform_arr: np.ndarray) -> np.ndarray:
        """Pure NumPy computation of log-Mel spectrogram."""
        # waveform_arr: [1, target_length] or [time]
        if waveform_arr.ndim == 2:
            sig = waveform_arr[0]
        else:
            sig = waveform_arr

        # Framing
        win = np.hanning(self.n_fft).astype(np.float32)
        n_frames = 1 + (len(sig) - self.n_fft) // self.hop_length
        if n_frames <= 0:
            # Handle edge case where signal < n_fft
            sig = np.pad(sig, (0, self.n_fft - len(sig) + self.hop_length), "constant")
            n_frames = 1 + (len(sig) - self.n_fft) // self.hop_length

        frames = np.zeros((n_frames, self.n_fft), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = sig[start : start + self.n_fft] * win

        # Real FFT & power spectrum
        fft_res = np.fft.rfft(frames, n=self.n_fft)
        power_spectrum = (np.abs(fft_res) ** 2) / float(self.n_fft)

        # Mel filterbank dot product -> [n_frames, n_mels]
        mel_spec = np.dot(power_spectrum, self.mel_filters.T)
        mel_spec = np.maximum(mel_spec, 1e-10)

        # Amplitude to dB
        log_mel = 10.0 * np.log10(mel_spec)
        # Transpose to [n_mels, n_frames]
        log_mel = log_mel.T

        # Standardisation (zero mean, unit variance)
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-6
        norm_log_mel = (log_mel - mean) / std

        # Shape [1, n_mels, n_frames]
        return norm_log_mel[np.newaxis, :, :].astype(np.float32)

    def compute_log_mel(self, waveform: Any) -> Any:
        """Computes standardized log-Mel spectrogram.

        Returns [1, n_mels=64, time_frames=94]
        """
        if HAS_TORCH and isinstance(waveform, torch.Tensor) and self.mel_transform is not None:
            try:
                if waveform.dim() == 2:
                    mel_spec = self.mel_transform(waveform)
                    log_mel = self.amplitude_to_db(mel_spec)
                elif waveform.dim() == 3:
                    mel_spec = self.mel_transform(waveform.squeeze(1))
                    log_mel = self.amplitude_to_db(mel_spec).unsqueeze(1)
                else:
                    mel_spec = self.mel_transform(waveform)
                    log_mel = self.amplitude_to_db(mel_spec)

                mean = log_mel.mean()
                std = log_mel.std() + 1e-6
                return (log_mel - mean) / std
            except Exception:
                pass  # Fall back to numpy

        # Fallback to pure numpy
        if HAS_TORCH and isinstance(waveform, torch.Tensor):
            arr = waveform.detach().cpu().numpy()
            res_arr = self.compute_log_mel_numpy(arr)
            return torch.from_numpy(res_arr)
        else:
            arr = np.array(waveform, dtype=np.float32)
            return self.compute_log_mel_numpy(arr)

    def add_background_noise(
        self,
        waveform: Any,
        noise: Optional[Any] = None,
        snr_db: float = 15.0,
    ) -> Any:
        """Injects additive noise at specified SNR (dB)."""
        is_tensor = HAS_TORCH and isinstance(waveform, torch.Tensor)
        arr = waveform.detach().cpu().numpy().copy() if is_tensor else np.array(waveform, dtype=np.float32).copy()

        sig_power = np.mean(arr ** 2)
        if noise is None:
            noise_arr = np.random.normal(0, 1.0, arr.shape).astype(np.float32)
        else:
            noise_arr = noise.detach().cpu().numpy().copy() if (HAS_TORCH and isinstance(noise, torch.Tensor)) else np.array(noise, dtype=np.float32)

        noise_power = np.mean(noise_arr ** 2)
        if noise_power == 0:
            return waveform

        snr_linear = 10.0 ** (snr_db / 10.0)
        target_noise_power = sig_power / snr_linear
        scale = np.sqrt(target_noise_power / (noise_power + 1e-8))

        noisy = arr + scale * noise_arr
        max_val = np.max(np.abs(noisy))
        if max_val > 1.0:
            noisy = noisy / max_val

        if is_tensor:
            return torch.from_numpy(noisy.astype(np.float32))
        return noisy.astype(np.float32)

    def apply_spec_augment(
        self,
        log_mel: Any,
        time_mask_param: int = 10,
        freq_mask_param: int = 8,
    ) -> Any:
        """Applies SpecAugment masking in either Torch or NumPy."""
        is_tensor = HAS_TORCH and isinstance(log_mel, torch.Tensor)
        arr = log_mel.detach().cpu().numpy().copy() if is_tensor else np.array(log_mel, dtype=np.float32).copy()

        # Frequency masking
        n_mels = arr.shape[-2]
        f_len = np.random.randint(0, freq_mask_param)
        f_start = np.random.randint(0, max(1, n_mels - f_len))
        arr[..., f_start : f_start + f_len, :] = 0.0

        # Time masking
        n_time = arr.shape[-1]
        t_len = np.random.randint(0, time_mask_param)
        t_start = np.random.randint(0, max(1, n_time - t_len))
        arr[..., :, t_start : t_start + t_len] = 0.0

        if is_tensor:
            return torch.from_numpy(arr)
        return arr

    def generate_synthetic_cue(self, cue_type: str, duration: Optional[float] = None) -> Any:
        """Generates realistic synthetic audio waveforms for unit testing and offline smoke tests."""
        dur = duration or self.target_duration
        num_samples = int(self.sample_rate * dur)
        t = np.linspace(0, dur, num_samples, endpoint=False)

        if cue_type == "background_normal":
            white_noise = np.random.normal(0, 0.05, num_samples)
            hum = 0.02 * np.sin(2 * np.pi * 50 * t)
            traffic_lfo = 0.03 * np.sin(2 * np.pi * 0.3 * t)
            signal = white_noise + hum + traffic_lfo

        elif cue_type == "screaming_shouting":
            f0 = 950 + 150 * np.sin(2 * np.pi * 6.0 * t)
            phase0 = 2 * np.pi * np.cumsum(f0) / self.sample_rate
            h1 = 0.40 * np.sin(phase0)
            h2 = 0.30 * np.sin(2 * phase0)
            h3 = 0.20 * np.sin(3 * phase0)
            roughness = 0.15 * np.random.normal(0, 0.1, num_samples)
            env = np.clip(np.sin(np.pi * (t / dur)), 0, 1) ** 0.5
            signal = (h1 + h2 + h3 + roughness) * env

        elif cue_type == "alarm_siren":
            siren_freq = 800 + 350 * np.sin(2 * np.pi * 1.2 * t)
            phase = 2 * np.pi * np.cumsum(siren_freq) / self.sample_rate
            carrier = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase)
            signal = carrier

        elif cue_type == "glass_breaking":
            t_decay = np.exp(-12.0 * t)
            shatter = np.random.normal(0, 0.5, num_samples) * t_decay
            high_ring = 0.3 * np.sin(2 * np.pi * 4200 * t) * t_decay
            high_ring2 = 0.2 * np.sin(2 * np.pi * 5800 * t) * t_decay
            signal = shatter + high_ring + high_ring2

        elif cue_type == "impact_crash":
            t_decay = np.exp(-8.0 * t)
            impulse = 0.6 * np.sin(2 * np.pi * 90 * t) * t_decay
            crash_noise = np.random.normal(0, 0.4, num_samples) * np.exp(-5.0 * t)
            signal = impulse + crash_noise

        else:
            raise ValueError(f"Unknown synthetic cue type: {cue_type}")

        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = 0.85 * (signal / max_val)

        arr = signal.astype(np.float32)[np.newaxis, :]
        return self.load_and_standardise(arr)
