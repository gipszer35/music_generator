import sys
import torch
import torchaudio
import os
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from IPython.display import Audio, display
import os
import datetime
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import numpy as np
from collections import deque
from dataclasses import dataclass


def is_colab():
    return "COLAB_GPU" in os.environ


@dataclass
class Config:
    root_dir: str
    work_dir: str
    batch_size: int

    sr: int = 8000
    seconds: int = 1
    epochs: int = 2000000
    timesteps: int = 400
    audio_dir = "nsynth-test/audio"
    lr = 5e-5

    @property
    def out_dir(self):
        return os.path.join(self.work_dir, "music_out")

    @property
    def model_file(self):
        return os.path.join(self.out_dir, "dit_wave.pt")

    @property
    def clip_len(self):
        return int(config.sr * self.seconds)


def create_config() -> Config:
    if is_colab():
        if not os.path.ismount("/content/drive"):
            from google.colab import drive

            drive.mount("/content/drive")

        root_dir = "/content/drive/MyDrive/"
        work_dir = root_dir + "MusicGenerator/"
        batch_size = 40
    else:
        root_dir = "./"
        work_dir = root_dir
        batch_size = 2

    return Config(root_dir=root_dir, work_dir=work_dir, batch_size=batch_size)


class DiffusionSchedule:
    def __init__(self, device):
        self.betas = my.cosine_beta_schedule(config.timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bars[t].view(-1, 1, 1)

    def get_alpha(self, t):
        return self.alphas[t].view(-1, 1, 1)


config = create_config()
sys.path.append(config.root_dir)
import my_common as my

schedule = DiffusionSchedule(my.DEVICE)
logger = my.create_logger()


def ensure_dir(path):
    if os.path.exists(path):
        logger.info(f"Directory already exists: {path}")
    else:
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


url = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
)

if not os.path.exists("nsynth-test"):
    !wget $url
    !tar -xzf nsynth-test.jsonwav.tar.gz

class NSynthSubset(Dataset):
    def __init__(self, root, instrument="keyboard"):
        self.paths = []
        for f in os.listdir(root):
            if f.endswith(".wav"):
                if instrument in f:
                    self.paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wav, sr0 = torchaudio.load(path)
        if wav.shape[0] > 1:
            # Convert to mono
            wav = wav.mean(dim=0, keepdim=True)
        if sr0 != config.sr:
            wav = torchaudio.transforms.Resample(sr0, config.sr)(wav)
        if wav.shape[1] > config.clip_len:
            start = random.randint(0, wav.shape[1] - config.clip_len)
            wav = wav[:, start : start + config.clip_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, config.clip_len - wav.shape[1]))
        if wav.abs().max() > 0:
            wav = wav / (wav.abs().max() + 1e-7)  # prevents divide-by-zero
        return wav


class SimpleUNet(nn.Module):
    def __init__(self, time_hidden_size=256):
        super().__init__()

        self.time_embedder = my.TimestepEmbedder(hidden_size=time_hidden_size)

        c1, c2, c3, c4, c5, c6, c7 = 64, 128, 256, 384, 512, 512, 512
        k = 7
        p = k // 2

        self.enc1 = nn.Conv1d(1, c1, k, padding=p)
        self.gn1 = nn.GroupNorm(8, c1)

        self.enc2 = nn.Conv1d(c1, c2, k, stride=2, padding=p)
        self.gn2 = nn.GroupNorm(8, c2)

        self.enc3 = nn.Conv1d(c2, c3, k, stride=2, padding=p)
        self.gn3 = nn.GroupNorm(8, c3)

        self.enc4 = nn.Conv1d(c3, c4, k, stride=2, padding=p)
        self.gn4 = nn.GroupNorm(8, c4)

        self.enc5 = nn.Conv1d(c4, c5, k, stride=2, padding=p)
        self.gn5 = nn.GroupNorm(8, c5)

        self.enc6 = nn.Conv1d(c5, c6, k, stride=2, padding=p)
        self.gn6 = nn.GroupNorm(8, c6)

        self.enc7 = nn.Conv1d(c6, c7, k, stride=2, padding=p)
        self.gn7 = nn.GroupNorm(8, c7)

        # Bottleneck
        self.bot = nn.Conv1d(c7, c7, k, padding=3, dilation=1)

        # Decoder
        self.dec7 = nn.ConvTranspose1d(c7, c6, 4, stride=2, padding=1)
        self.dec6 = nn.ConvTranspose1d(c6, c5, 4, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose1d(c5, c4, 4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose1d(c4, c3, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose1d(c3, c2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1)

        self.out = nn.Sequential(nn.Conv1d(c1, 1, k, padding=p), nn.Tanh())

        self.scale1, self.shift1 = nn.Linear(time_hidden_size, c1), nn.Linear(
            time_hidden_size, c1
        )
        self.scale2, self.shift2 = nn.Linear(time_hidden_size, c2), nn.Linear(
            time_hidden_size, c2
        )
        self.scale3, self.shift3 = nn.Linear(time_hidden_size, c3), nn.Linear(
            time_hidden_size, c3
        )
        self.scale4, self.shift4 = nn.Linear(time_hidden_size, c4), nn.Linear(
            time_hidden_size, c4
        )
        self.scale5, self.shift5 = nn.Linear(time_hidden_size, c5), nn.Linear(
            time_hidden_size, c5
        )
        self.scale6, self.shift6 = nn.Linear(time_hidden_size, c6), nn.Linear(
            time_hidden_size, c6
        )
        self.scale7, self.shift7 = nn.Linear(time_hidden_size, c7), nn.Linear(
            time_hidden_size, c7
        )

    def film(self, x, scale, shift, t_emb):
        scale = scale(t_emb)[:, :, None]
        shift = shift(t_emb)[:, :, None]
        return x * (1 + scale) + shift

    def match_and_add(self, dec_x, enc_x):
        """Fixes potential odd dimension mismatches in the skip connection"""
        if dec_x.shape[-1] != enc_x.shape[-1]:
            dec_x = dec_x[..., : enc_x.shape[-1]]
        return torch.relu(dec_x + enc_x)

    def forward(self, x, t):
        t_emb = self.time_embedder(t)

        # Encode path
        x1 = torch.relu(
            self.gn1(self.film(self.enc1(x), self.scale1, self.shift1, t_emb))
        )
        x2 = torch.relu(
            self.gn2(self.film(self.enc2(x1), self.scale2, self.shift2, t_emb))
        )
        x3 = torch.relu(
            self.gn3(self.film(self.enc3(x2), self.scale3, self.shift3, t_emb))
        )
        x4 = torch.relu(
            self.gn4(self.film(self.enc4(x3), self.scale4, self.shift4, t_emb))
        )
        x5 = torch.relu(
            self.gn5(self.film(self.enc5(x4), self.scale5, self.shift5, t_emb))
        )
        x6 = torch.relu(
            self.gn6(self.film(self.enc6(x5), self.scale6, self.shift6, t_emb))
        )
        x7 = torch.relu(
            self.gn7(self.film(self.enc7(x6), self.scale7, self.shift7, t_emb))
        )

        # Bottleneck
        b = torch.relu(self.bot(x7))

        # Decode path
        d7 = self.match_and_add(self.dec7(b), x6)
        d6 = self.match_and_add(self.dec6(d7), x5)
        d5 = self.match_and_add(self.dec5(d6), x4)
        d4 = self.match_and_add(self.dec4(d5), x3)
        d3 = self.match_and_add(self.dec3(d4), x2)
        d2 = self.match_and_add(self.dec2(d3), x1)

        return self.out(d2)


class CheckpointManager:
    def save_checkpoint(self, model, optimizer, path=config.model_file):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, model, optimizer=None, path=config.model_file):
        checkpoint = torch.load(path, map_location=my.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from: {path}")

        if optimizer and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state.")
            except Exception as e:
                logger.info(f"Optimizer state not loaded: {e}")

    def load_model(self):
        model = SimpleUNet().to(my.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        if os.path.exists(config.model_file):
            self.load_checkpoint(model, optimizer)
        else:
            logger.info("No checkpoint found — initialized new model and optimizer.")
        my.print_parameter_summary(model)
        return model, optimizer


class AudioVisualizer:

    def show_generated_sample(self, generated_sample):
        logger.info("Generating sample...")
        display(Audio(generated_sample, rate=config.sr))

    def show_dataset_sample(self, dataset):
        index = random.randint(0, len(dataset) - 1)
        wav = dataset[index]
        if isinstance(wav, tuple):  # in case your Dataset returns (wav, label)
            wav = wav[0]
        logger.info(f"Dataset sample at index {index}:")
        display(Audio(wav.squeeze().numpy(), rate=config.sr))

    def show_samples(self, generated_sample, dataset):
        logger.info(f"Date:{datetime.datetime.now()}")
        self.show_generated_sample(generated_sample)
        self.show_dataset_sample(dataset)

    def show_comparison_plot(self, clean, noisy, pred_denoised, sr=config.sr, title=""):
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


class DitWaveTrainer:
    def __init__(self):
        self.dataset = NSynthSubset(config.audio_dir)
        self.loader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=True
        )
        self.checkpoint_manager = CheckpointManager()
        self.model, self.optimizer = self.checkpoint_manager.load_model()
        self.mse = nn.MSELoss()
        self.loss_history = deque(maxlen=1000)
        self.visualizer = AudioVisualizer()

    @torch.no_grad()
    def sample(self, model, steps, shape):
        x_t = torch.randn(shape).to(my.DEVICE)
        for t in reversed(range(steps)):
            t_tensor = torch.full((shape[0],), t, device=my.DEVICE, dtype=torch.long)
            pred_noise = model(x_t, t_tensor)

            alpha = schedule.get_alpha(t)
            alpha_bar = schedule.get_alpha_bar(t)
            beta = schedule.betas[t].view(-1, 1, 1)

            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            x_t = (1 / alpha.sqrt()) * (
                x_t - (1 - alpha) / (1 - alpha_bar).sqrt() * pred_noise
            ) + beta.sqrt() * noise
        return x_t

    def update_avg_loss(self, loss_value):
        self.loss_history.append(loss_value)
        return sum(self.loss_history) / len(self.loss_history)

    def show_prot_graphs(self, model, epoch, i, x0, t, x_noisy):
        model.eval()
        with torch.no_grad():
            sample_x0 = x0[0:1]
            sample_t = t[0:1]
            x_t = x_noisy[0:1]
            predicted_noise = model(x_t, sample_t)
            a_bar = schedule.get_alpha_bar(sample_t)

            denoised = (x_t - torch.sqrt(1 - a_bar) * predicted_noise) / torch.sqrt(
                a_bar
            )
            denoised.clamp_(-1, 1)

            self.visualizer.show_comparison_plot(
                clean=sample_x0,
                noisy=x_t,
                pred_denoised=denoised,
                title=f"Epoch {epoch}, Step {i} current t={t[0:1].item()}",
            )
        model.train()

    def train(self):
        step = 0
        self.model.train()

        logger.info(
            f"Training started! Number of data samples: {len(self.dataset)} | Batches: {len(self.loader)}"
        )

        for epoch in range(config.epochs):
            for _, x0 in enumerate(self.loader):
                x0 = x0.to(my.DEVICE)

                t = torch.randint(
                    0, config.timesteps, (x0.shape[0],), device=my.DEVICE
                ).long()

                noise = torch.randn_like(x0).to(my.DEVICE)

                a_bar = schedule.get_alpha_bar(t)
                x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

                self.optimizer.zero_grad(set_to_none=True)

                pred = self.model(x_t, t)
                loss = self.mse(pred, noise)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                avg_loss = self.update_avg_loss(loss.item())

                if step % 50 == 0:
                    self.model.eval()

                    self.checkpoint_manager.save_checkpoint(self.model, self.optimizer)
                    logger.info(
                        f"Epoch {epoch} "
                        f"Step {step} "
                        f"Loss: {loss.item():.4f} "
                        f"Avg Loss: {avg_loss:.4f}"
                    )
                    generated_sample = (
                        self.sample(self.model, config.timesteps, (1, 1, config.clip_len))
                        .cpu()
                        .squeeze()
                        .numpy()
                    )
                    self.visualizer.show_samples(generated_sample, self.dataset)
                    self.show_prot_graphs(self.model, epoch, step, x0, t, x_t)

                    self.model.train()
                step += 1


def train():
    ensure_dir(config.out_dir)
    trainer = DitWaveTrainer()
    trainer.train()


if __name__ == "__main__":
    train()
