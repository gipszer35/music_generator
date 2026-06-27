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
from functools import lru_cache


@dataclass
class AudioCodecComponents:
    model: EncodecModel
    processor: AutoProcessor


class AudioCodecFactory:
    @staticmethod
    @lru_cache(maxsize=None)
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


class NSynthSubset(Dataset):
    URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
    ARCHIVE = "nsynth-test.jsonwav.tar.gz"
    DATASET_DIR = "nsynth-test"

    def __init__(self, sample_rate: int, clip_len, out_dir):
        self.sample_rate = sample_rate
        self.clip_len = clip_len
        self.out_dir = out_dir

        NSynthSubset._prepare(self.out_dir)

        audio_dir = os.path.join(self.out_dir, self.DATASET_DIR, "audio")

        self.paths = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".wav")
        ]

    @classmethod
    def _prepare(cls, out_dir):
        archive = os.path.join(out_dir, cls.ARCHIVE)
        dataset_dir = os.path.join(out_dir, cls.DATASET_DIR)

        if not os.path.isdir(dataset_dir):
            if not os.path.isfile(archive):
                print("Downloading NSynth...")
                urllib.request.urlretrieve(cls.URL, archive)

            print("Extracting NSynth...")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(out_dir)

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

    def load_model(self, model):

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

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
