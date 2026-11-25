"""Upgraded Audio Preprocessing Module with 128-Mel Resolution and Mixup Augmentation.
Owner: Debajyoti (Audio ML Lead)

Upgrades for High-Accuracy Real-World Performance:
- 128 Mel filterbank bins (captures granular vocal formants & high-frequency transients)
- Mixup Audio Augmentation (blends environmental soundscapes to simulate real Kolkata streets)
- SpecAugment with adaptive time/frequency masking
- Dual PyTorch/Torchaudio and NumPy runtime support
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union

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
    n_mels: int = 128,
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

    # Slaney-style area normalisation
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
        n_mels: int = 128,
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
        if HAS_TORCH and torch is not None:
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
        if HAS_TORCH and isinstance(audio_input, torch.Tensor):
            arr = audio_input.detach().cpu().numpy()
            return_tensor = True
        elif isinstance(audio_input, np.ndarray):
            arr = audio_input.copy()
            return_tensor = HAS_TORCH
        elif isinstance(audio_input, str):
            # Try torchaudio first, fallback to wave
            loaded = False
            if HAS_TORCH and torchaudio is not None:
                try:
                    wf, sr = torchaudio.load(audio_input)
                    arr = wf.numpy()
                    input_sample_rate = sr
                    return_tensor = True
                    loaded = True
                except Exception:
                    pass

            if not loaded:
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
        """Pure NumPy computation of log-Mel spectrogram with current n_mels."""
        if waveform_arr.ndim == 2:
            sig = waveform_arr[0]
        else:
            sig = waveform_arr

        win = np.hanning(self.n_fft).astype(np.float32)
        n_frames = 1 + (len(sig) - self.n_fft) // self.hop_length
        if n_frames <= 0:
            sig = np.pad(sig, (0, self.n_fft - len(sig) + self.hop_length), "constant")
            n_frames = 1 + (len(sig) - self.n_fft) // self.hop_length

        frames = np.zeros((n_frames, self.n_fft), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = sig[start : start + self.n_fft] * win

        fft_res = np.fft.rfft(frames, n=self.n_fft)
        power_spectrum = (np.abs(fft_res) ** 2) / float(self.n_fft)

        mel_spec = np.dot(power_spectrum, self.mel_filters.T)
        mel_spec = np.maximum(mel_spec, 1e-10)

        log_mel = 10.0 * np.log10(mel_spec).T

        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-6
        norm_log_mel = (log_mel - mean) / std

        return norm_log_mel[np.newaxis, :, :].astype(np.float32)

    def compute_log_mel(self, waveform: Any) -> Any:
        """Computes standardized log-Mel spectrogram [1, n_mels=128, time_frames=94]."""
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
                pass

        if HAS_TORCH and isinstance(waveform, torch.Tensor):
            arr = waveform.detach().cpu().numpy()
            res_arr = self.compute_log_mel_numpy(arr)
            return torch.from_numpy(res_arr)
        else:
            arr = np.array(waveform, dtype=np.float32)
            return self.compute_log_mel_numpy(arr)

    def apply_mixup(
        self,
        waveform_a: Any,
        waveform_b: Any,
        alpha: float = 0.4,
    ) -> Tuple[Any, float]:
        """Applies Mixup augmentation between two sound clips.

        Simulates real-world acoustic layering (e.g. scream superimposed on street noise).
        Returns (mixed_waveform, lambda_factor).
        """
        is_tensor = HAS_TORCH and isinstance(waveform_a, torch.Tensor)
        a = waveform_a.detach().cpu().numpy() if is_tensor else np.array(waveform_a, dtype=np.float32)
        b = waveform_b.detach().cpu().numpy() if is_tensor else np.array(waveform_b, dtype=np.float32)

        # Draw lambda from Beta distribution
        lam = float(np.random.beta(alpha, alpha))
        # Keep lam between 0.2 and 0.8 to ensure meaningful blending
        lam = max(0.2, min(0.8, lam))

        mixed = lam * a + (1.0 - lam) * b
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val

        if is_tensor:
            return torch.from_numpy(mixed.astype(np.float32)), lam
        return mixed.astype(np.float32), lam

    def add_background_noise(
        self,
        waveform: Any,
        noise: Optional[Any] = None,
        snr_db: float = 15.0,
    ) -> Any:
        """Injects additive ambient noise at specified SNR (dB)."""
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
        time_mask_param: int = 12,
        freq_mask_param: int = 16,
    ) -> Any:
        """Applies SpecAugment masking in either Torch or NumPy."""
        is_tensor = HAS_TORCH and isinstance(log_mel, torch.Tensor)
        arr = log_mel.detach().cpu().numpy().copy() if is_tensor else np.array(log_mel, dtype=np.float32).copy()

        n_mels = arr.shape[-2]
        f_len = np.random.randint(0, freq_mask_param)
        f_start = np.random.randint(0, max(1, n_mels - f_len))
        arr[..., f_start : f_start + f_len, :] = 0.0

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
            # Realistic physical acoustic synthesis of shattering glass:
            # 1. Primary fracture crack (wideband impulse with sharp envelope)
            t_crack = np.exp(-18.0 * t)
            crack_burst = np.random.normal(0, 0.5, num_samples) * t_crack

            # 2. Resonant high-frequency crystal/glass ringing (modal vibrations at 3.8k, 5.2k, 6.8k, 7.5k Hz)
            modal_ring = (
                0.35 * np.sin(2 * np.pi * 3800 * t) * np.exp(-10.0 * t)
                + 0.30 * np.sin(2 * np.pi * 5200 * t) * np.exp(-14.0 * t)
                + 0.25 * np.sin(2 * np.pi * 6800 * t) * np.exp(-16.0 * t)
                + 0.15 * np.sin(2 * np.pi * 7500 * t) * np.exp(-20.0 * t)
            )

            # 3. Cascading glass shard bounces & fragments (tinkling across 0.1s to 1.2s)
            shard_tinkle = np.zeros(num_samples, dtype=np.float32)
            shard_offsets = [0.12, 0.28, 0.45, 0.72, 1.05]
            shard_freqs = [4400, 5800, 6300, 4900, 7200]
            for off, freq in zip(shard_offsets, shard_freqs):
                shard_idx = int(off * self.sample_rate)
                if shard_idx < num_samples:
                    dt = t[: num_samples - shard_idx]
                    shard_env = np.exp(-28.0 * dt)
                    shard_sig = (
                        np.random.normal(0, 0.25, len(dt))
                        + 0.35 * np.sin(2 * np.pi * freq * dt)
                    ) * shard_env
                    shard_tinkle[shard_idx:] += shard_sig.astype(np.float32)

            signal = crack_burst + modal_ring + shard_tinkle

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
