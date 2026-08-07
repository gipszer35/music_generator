import sys
import os
import torch
import torchaudio
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from IPython.display import Audio, display
import datetime
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import numpy as np
from collections import deque
from dataclasses import dataclass
import tarfile
import urllib.request
from transformers import EncodecModel, AutoProcessor


@dataclass
class AudioCodecComponents:
    model: EncodecModel
    processor: AutoProcessor


class AudioCodecFactory:
    @staticmethod
    def create(device, model_name: str = "facebook/encodec_24khz"):
        model = EncodecModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)

        model.eval()

        return AudioCodecComponents(model=model, processor=processor)


def ensure_dir(path, logger):
    if os.path.exists(path):
        logger.info(f"Directory already exists: {path}")
    else:
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


class MusicDataset(Dataset):
    def __init__(self, sample_rate: int, clip_len, out_dir, data_dir="training-data"):
        self.sample_rate = sample_rate
        self.clip_len = clip_len
        self.out_dir = out_dir
        self.data_dir = data_dir

        audio_dir = os.path.join(self.out_dir, self.data_dir, "audio")

        self.paths = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".wav")
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.paths[idx])

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to codec sample rate
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                self.sample_rate,
            )

        # Random crop / pad
        if wav.shape[1] > self.clip_len:
            start = random.randint(0, wav.shape[1] - self.clip_len)
            wav = wav[:, start : start + self.clip_len]
        else:
            wav = torch.nn.functional.pad(
                wav,
                (0, self.clip_len - wav.shape[1]),
            )

        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        return wav


class NSynthSubset(MusicDataset):
    URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
    ARCHIVE = "nsynth-test.jsonwav.tar.gz"
    DATA_DIR = "nsynth-test"

    def __init__(self, sample_rate: int, clip_len, out_dir):
        self._prepare(out_dir)
        super().__init__(sample_rate, clip_len, out_dir, self.DATA_DIR)

    def _prepare(self, out_dir):
        archive = os.path.join(out_dir, self.ARCHIVE)
        dataset_dir = os.path.join(out_dir, self.DATA_DIR)

        if not os.path.isdir(dataset_dir):
            if not os.path.isfile(archive):
                print("Downloading NSynth...")
                urllib.request.urlretrieve(self.URL, archive)

            print("Extracting NSynth...")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(out_dir)


class DACCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        raw_audio = [x.squeeze(0).numpy() for x in batch]

        return self.processor(
            raw_audio=raw_audio,
            sampling_rate=self.processor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )


class CheckpointManager:
    def __init__(self, model_file, lr, device, logger):
        self.model_file = model_file
        self.lr = lr
        self.device = device
        self.logger = logger

    def save_checkpoint(self, model, optimizer):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            self.model_file,
        )

    def load_checkpoint(self, model, optimizer=None):
        checkpoint = torch.load(self.model_file, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info(f"Loaded model from: {self.model_file}")

        if optimizer and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.logger.info("Loaded optimizer state.")
            except Exception as e:
                self.logger.info(f"Optimizer state not loaded: {e}")

    def load_model(self, model, optimizer):

        if os.path.exists(self.model_file):
            self.load_checkpoint(model, optimizer)
        else:
            self.logger.info(
                "No checkpoint found — initialized new model and optimizer."
            )
        return model, optimizer


class AudioVisualizer:
    def __init__(self, logger):
        self.logger = logger

    def show_generated_sample(self, generated_sample, sr):
        self.logger.info("Generating sample...")
        display(Audio(generated_sample, rate=sr))

    def show_dataset_sample(self, dataset, sr):
        index = random.randint(0, len(dataset) - 1)
        wav = dataset[index]
        if isinstance(wav, tuple):  # Dataset returns (wav, label)
            wav = wav[0]
        self.logger.info(f"Dataset sample at index {index}:")
        display(Audio(wav.squeeze().numpy(), rate=sr))

    def show_samples(self, generated_sample, dataset, sr):
        self.logger.info(f"Date:{datetime.datetime.now()}")
        self.show_generated_sample(generated_sample, sr)
        self.show_dataset_sample(dataset, sr)

    def show_comparison(self, clean, noisy, pred_denoised, sr, title=""):
        def plot_waveform(ax, signal, label):
            ax.plot(signal.squeeze().cpu().numpy())
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")

        def plot_mel_spectrogram(ax, signal, label):
            mel_transform = T.MelSpectrogram(
                sample_rate=sr, n_fft=1024, hop_length=256, n_mels=64
            ).to(signal.device)
            mel_spec = mel_transform(signal)
            mel_spec_db = T.AmplitudeToDB()(mel_spec)
            ax.imshow(
                mel_spec_db.squeeze().cpu().numpy(),
                aspect="auto",
                origin="lower",
                cmap="magma",
            )
            ax.set_title(label)
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel Bin")

        def plot_stft_spectrogram(ax, signal, label):
            n_fft = 1024
            hop_length = 256

            # Compute STFT (shape: [freq_bins, time_bins])
            spec = torch.stft(
                signal.squeeze(0),
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
            )
            spec_db = spec.abs().pow(2).log1p()

            # Convert to numpy (now shape is [freq_bins, time_bins])
            spec_db_np = spec_db.squeeze(0).cpu().numpy()

            # Frequency and time axes
            freq = np.fft.rfftfreq(n_fft, d=1 / sr)  # freq_bins
            time = np.linspace(
                0, signal.shape[-1] / sr, spec_db_np.shape[1]
            )  # time_bins

            ax.imshow(
                spec_db_np,
                aspect="auto",
                origin="lower",
                extent=[time[0], time[-1], freq[0], freq[-1]],
                cmap="plasma",
            )

            ax.set_title(label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_yscale("log")

            # Optional: Set clearer frequency ticks
            ax.set_yticks([50, 100, 200, 400, 800, 1600, 3200, 6400])
            ax.get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{int(y)}")
            )

        _, axs = plt.subplots(3, 3, figsize=(15, 9))

        plot_waveform(axs[0, 0], clean, "Clean Audio")
        plot_waveform(axs[0, 1], noisy, "Noisy Input")
        plot_waveform(axs[0, 2], pred_denoised, "Predicted Denoised")

        plot_mel_spectrogram(axs[1, 0], clean, "Clean Mel Spectrogram")
        plot_mel_spectrogram(axs[1, 1], noisy, "Noisy Mel Spectrogram")
        plot_mel_spectrogram(axs[1, 2], pred_denoised, "Denoised Mel Spectrogram")

        plot_stft_spectrogram(axs[2, 0], clean, "Clean STFT Spectrogram")
        plot_stft_spectrogram(axs[2, 1], noisy, "Noisy STFT Spectrogram")
        plot_stft_spectrogram(axs[2, 2], pred_denoised, "Denoised STFT Spectrogram")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class AudioFidelityEvaluator:
    """
    A non-trainable utility class that bundles all multi-resolution spectral
    loss functions for high-fidelity audio training.
    """

    def __init__(
        self,
        encodec,
        sample_rate=44100,
        device="cpu",
        mse_weight=1.0,
        stft_weight=1.0,
        mel_weight=1.0,
    ):
        self.device = device
        self.encodec = encodec
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight

        # Multi-resolution STFT configurations
        self.stft_fft_sizes = [512, 1024, 2048]
        self.stft_hop_sizes = [50, 120, 240]
        self.stft_win_sizes = [240, 600, 1200]

        # Pre-instantiate Mel-spectrogram transforms directly on the target device
        self.mel_transforms = [
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                n_mels=80,
            ).to(device),
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                n_mels=128,
            ).to(device),
        ]

    def _stft_loss(self, pred, target):
        """Calculates multi-resolution spectral convergence and log magnitude loss."""
        stft_loss_val = 0.0
        for f, h, w in zip(
            self.stft_fft_sizes, self.stft_hop_sizes, self.stft_win_sizes
        ):
            window = torch.hann_window(w).to(pred.device)

            p_stft = torch.stft(
                pred.squeeze(1),
                n_fft=f,
                hop_length=h,
                win_length=w,
                window=window,
                return_complex=True,
            )
            t_stft = torch.stft(
                target.squeeze(1),
                n_fft=f,
                hop_length=h,
                win_length=w,
                window=window,
                return_complex=True,
            )

            p_mag = torch.abs(p_stft) + 1e-7
            t_mag = torch.abs(t_stft) + 1e-7

            sc_loss = torch.norm(t_mag - p_mag, p="fro") / (
                torch.norm(t_mag, p="fro") + 1e-9
            )
            log_mag_loss = torch.mean(torch.abs(torch.log(t_mag) - torch.log(p_mag)))

            stft_loss_val += sc_loss + log_mag_loss

        return stft_loss_val / len(self.stft_fft_sizes)

    def _mel_loss(self, pred, target):
        mel_loss_val = 0.0
        for mel_transform in self.mel_transforms:
            p_mel = mel_transform(pred.squeeze(1))
            t_mel = mel_transform(target.squeeze(1))

            # Convert to log-scale to normalize loss magnitude
            p_mel_log = torch.log(p_mel + 1e-5)
            t_mel_log = torch.log(t_mel + 1e-5)

            mel_loss_val += torch.mean(torch.abs(p_mel_log - t_mel_log))
        return mel_loss_val / len(self.mel_transforms)

    def compute(self, pred_noise, target_noise, z_t=None, a_bar=None, z_0=None):
        """
        Computes diffusion MSE + optional audio-domain perceptual losses.

        pred_noise: Predicted noise from diffusion model [B, 128, 128]
        target_noise: True noise [B, 128, 128]
        z_t: Noisy latent [B, 128, 128]
        a_bar: Cumulative alpha value [B, 1, 1]
        z_0: Clean latent [B, 128, 128]
        """

        def combine_losses(loss_mse, loss_stft, loss_mel):
            # Weighted contributions
            mse_term = self.mse_weight * loss_mse
            stft_term = self.stft_weight * loss_stft
            mel_term = self.mel_weight * loss_mel

            total = mse_term + stft_term + mel_term

            msg = (
                f"total_loss={total.item():.3f} = "
                f"{self.mse_weight:g}*mse={loss_mse.item():.3f} + "
                f"{self.stft_weight:g}*stft={loss_stft.item():.3f} + "
                f"{self.mel_weight:g}*mel={loss_mel.item():.3f} = "
                f"{mse_term.item():.3f} + {stft_term.item():.3f} + {mel_term.item():.3f}"
            )

            return total, msg

        # 1. Standard diffusion noise prediction loss
        loss_mse = torch.mean((pred_noise - target_noise) ** 2)

        # 2. Audio perceptual losses (only if latent context is available)
        if z_t is not None and a_bar is not None and z_0 is not None:

            sqrt_one_minus_a_bar = torch.sqrt(1.0 - a_bar)
            sqrt_a_bar = torch.sqrt(a_bar)

            # Predict clean latent z0 from predicted noise
            z_0_pred = (z_t - sqrt_one_minus_a_bar * pred_noise) / (sqrt_a_bar + 1e-9)

            batch_size = z_0.shape[0]
            perceptual_batch = max(batch_size // 16, 1)

            z0_pred_loss = z_0_pred[:perceptual_batch]
            z0_loss = z_0[:perceptual_batch]

            with torch.no_grad():
                audio_t = self.encodec.decoder(z0_loss)

            with torch.backends.cudnn.flags(enabled=False):
                audio_p = self.encodec.decoder(z0_pred_loss)

            if audio_p.dim() == 2:
                audio_p = audio_p.unsqueeze(1)

            if audio_t.dim() == 2:
                audio_t = audio_t.unsqueeze(1)

            # Spectral losses
            loss_stft = self._stft_loss(audio_p, audio_t)
            loss_mel = self._mel_loss(audio_p, audio_t)

            total_loss, loss_msg = combine_losses(loss_mse, loss_stft, loss_mel)

            return total_loss, loss_msg

        # Fallback: pure diffusion loss
        return loss_mse, None
