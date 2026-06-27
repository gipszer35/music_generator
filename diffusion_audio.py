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


def is_colab():
    return "COLAB_GPU" in os.environ


@dataclass
class Config:
    root_dir: str
    work_dir: str
    batch_size: int

    clip_len: int = 40960  # divide this by sample rate to get the lenght in sec
    latent_channels = 128
    downsampling_factor = 320
    epochs: int = 2_000_000
    timesteps: int = 700
    lr: float = 5e-5

    @property
    def out_dir(self):
        return os.path.join(self.work_dir, "music_out")

    @property
    def model_file(self):
        return os.path.join(self.out_dir, "diffusion_audio.pt")


def create_config() -> Config:
    if is_colab():
        if not os.path.ismount("/content/drive"):
            from google.colab import drive

            drive.mount("/content/drive")

        root_dir = "/content/drive/MyDrive/"
        work_dir = root_dir + "MusicGenerator/"
        batch_size = 384
    else:
        root_dir = "./"
        work_dir = root_dir
        batch_size = 3

    return Config(root_dir=root_dir, work_dir=work_dir, batch_size=batch_size)


class DiffusionSchedule:
    def __init__(self, device, timesteps):
        self.betas = my.cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bars[t].view(-1, 1, 1)

    def get_alpha(self, t):
        return self.alphas[t].view(-1, 1, 1)


config = create_config()
sys.path.append(config.root_dir)
sys.path.append(config.work_dir)
import my_common as my
import audio_common


schedule = DiffusionSchedule(my.DEVICE, config.timesteps)
logger = my.create_logger()


class UNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.act1(self.gn1(self.conv1(x)))
        # Inject time embedding into the feature maps
        h = h + self.mlp(t_emb).unsqueeze(-1)
        h = self.act2(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)


class DiffusionWave(nn.Module):
    def __init__(self, *, in_channels, model_channels, time_hidden_size):
        super().__init__()
        self.time_mlp = my.TimestepEmbedder(hidden_size=time_hidden_size)

        self.init_conv = nn.Conv1d(
            in_channels, model_channels, kernel_size=3, padding=1
        )

        self.down1 = UNetBlock1D(model_channels, model_channels, time_hidden_size)
        self.down1_pool = nn.Conv1d(
            model_channels, model_channels * 2, kernel_size=4, stride=2, padding=1
        )

        self.down2 = UNetBlock1D(
            model_channels * 2, model_channels * 2, time_hidden_size
        )
        self.down2_pool = nn.Conv1d(
            model_channels * 2, model_channels * 4, kernel_size=4, stride=2, padding=1
        )

        self.mid1 = UNetBlock1D(
            model_channels * 4, model_channels * 4, time_hidden_size
        )
        self.mid2 = UNetBlock1D(
            model_channels * 4, model_channels * 4, time_hidden_size
        )

        self.up2_unpool = nn.ConvTranspose1d(
            model_channels * 4, model_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.up2 = UNetBlock1D(model_channels * 4, model_channels * 2, time_hidden_size)

        self.up1_unpool = nn.ConvTranspose1d(
            model_channels * 2, model_channels, kernel_size=4, stride=2, padding=1
        )
        self.up1 = UNetBlock1D(model_channels * 2, model_channels, time_hidden_size)

        self.out_conv = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x1 = self.init_conv(x)
        x1 = self.down1(x1, t_emb)
        x2 = self.down1_pool(x1)

        x2 = self.down2(x2, t_emb)
        x3 = self.down2_pool(x2)

        x3 = self.mid1(x3, t_emb)
        x3 = self.mid2(x3, t_emb)

        x_up = self.up2_unpool(x3)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.up2(x_up, t_emb)

        x_up = self.up1_unpool(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.up1(x_up, t_emb)

        out = self.out_conv(x_up)

        return out


class DiffusionWaveTrainer:
    def __init__(self):
        audio_codec_components = audio_common.AudioCodecFactory.create(my.DEVICE)
        self.processor = audio_codec_components.processor
        self.encodec = audio_codec_components.model
        self.sr = self.processor.sampling_rate

        self.dataset = audio_common.NSynthSubset(
            sample_rate=self.sr, clip_len=config.clip_len, out_dir=config.out_dir
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=audio_common.DACCollator(self.processor),
        )

        self.checkpoint_manager = audio_common.CheckpointManager(
            model_file=config.model_file, lr=config.lr, device=my.DEVICE, logger=logger
        )
        model = DiffusionWave(
            in_channels=128, model_channels=256, time_hidden_size=512
        ).to(my.DEVICE)
        self.model, self.optimizer = self.checkpoint_manager.load_model(model)
        my.print_parameter_summary(self.model)
        self.mse = nn.MSELoss()
        self.loss_history = deque(maxlen=1000)
        self.visualizer = audio_common.AudioVisualizer(logger=logger)

    @torch.no_grad()
    def sample(self):
        shape = (
            1,
            config.latent_channels,
            config.clip_len // config.downsampling_factor,
        )
        x_t = torch.randn(shape).to(my.DEVICE)

        for t in reversed(range(config.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=my.DEVICE, dtype=torch.long)
            pred_noise = self.model(x_t, t_tensor)

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

        audio_waveform = self.encodec.decoder(x_t)

        return audio_waveform

    def update_avg_loss(self, loss_value):
        self.loss_history.append(loss_value)
        return sum(self.loss_history) / len(self.loss_history)

    def show_current_state(self, model, epoch, i, z_0, t, z_noisy):
        model.eval()
        with torch.no_grad():
            sample_z0 = z_0[0:1]
            sample_t = t[0:1]
            z_t = z_noisy[0:1]
            predicted_noise = model(z_t, sample_t)
            a_bar = schedule.get_alpha_bar(sample_t)

            denoised = (z_t - torch.sqrt(1 - a_bar) * predicted_noise) / torch.sqrt(
                a_bar
            )
            denoised.clamp_(-1, 1)

            self.visualizer.show_comparison(
                clean=self.encodec.decoder(sample_z0),
                noisy=self.encodec.decoder(z_t),
                pred_denoised=self.encodec.decoder(denoised),
                sr=self.sr,
                title=f"Epoch {epoch}, Step {i} current t={t[0:1].item()}",
            )
        model.train()

    def train(self):
        step = 0
        self.model.train()
        logger.info(
            f"Training started!\n"
            f"Number of data samples: {len(self.dataset)}\n"
            f"Batch Size: {config.batch_size}\n"
            f"Batches: {len(self.loader)}"
        )

        for epoch in range(config.epochs):
            for _, batch in enumerate(self.loader):
                waveform = batch["input_values"].to(my.DEVICE)

                with torch.no_grad():
                    z_0 = self.encodec.encoder(waveform)  # continuous latent

                t = torch.randint(0, config.timesteps, (z_0.size(0),), device=my.DEVICE)

                noise = torch.randn_like(z_0)

                a_bar = schedule.get_alpha_bar(t)
                z_t = torch.sqrt(a_bar) * z_0 + torch.sqrt(1 - a_bar) * noise
                self.optimizer.zero_grad(set_to_none=True)

                pred = self.model(z_t, t)
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
                    generated_sample = self.sample().cpu().squeeze().numpy()
                    self.visualizer.show_samples(
                        generated_sample, self.dataset, self.sr
                    )
                    self.show_current_state(self.model, epoch, step, z_0, t, z_t)

                    self.model.train()

                step += 1

        inputs = next(iter(self.loader))

        encoder_outputs = self.audio_codec_components.model.encode(
            inputs["input_values"], inputs["padding_mask"]
        )
        audio_codes = encoder_outputs.audio_codes
        audio_scales = encoder_outputs.audio_scales

        decoder_output = self.audio_codec_components.model.decode(
            audio_codes, audio_scales
        )
        reconstructed_audio = decoder_output.audio_values

        print("Original sound")
        display(
            Audio(
                inputs["input_values"][0],
                rate=self.audio_codec_components.processor.sampling_rate,
            )
        )

        print("\nGenerated sound")
        display(
            Audio(
                reconstructed_audio[0].detach().cpu().numpy(),
                rate=self.audio_codec_components.processor.sampling_rate,
            )
        )
        logger.info(f"Inputs shape: {inputs["input_values"].shape}")
        logger.info(f"outputs shape: {reconstructed_audio.shape}")


def train():
    audio_common.ensure_dir(config.out_dir, logger=logger)
    trainer = DiffusionWaveTrainer()
    trainer.train()


if __name__ == "__main__":
    train()
