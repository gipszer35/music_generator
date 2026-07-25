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
import torch.nn.functional as F


def is_colab():
    return "COLAB_GPU" in os.environ


@dataclass
class Config:
    root_dir: str
    work_dir: str
    batch_size: int

    clip_len: int = 40960  # divide this by sample rate to get the length in sec
    latent_channels = 128
    downsampling_factor = 320
    epochs: int = 2_000_000
    lr: float = 5e-5

    n_embed: int = 512  # Embedding dimension for the transformer
    block_size: int = 2048  # Maximum sequence length (S * K) the transformer can accept
    n_head: int = 8  # Number of attention heads
    n_layer: int = 6  # Number of transformer blocks
    dropout: float = 0.1  # Dropout rate

    @property
    def out_dir(self):
        return os.path.join(self.work_dir, "music_out")

    @property
    def model_file(self):
        return os.path.join(self.out_dir, "transformer_audio.pt")


def create_config() -> Config:
    if is_colab():
        if not os.path.ismount("/content/drive"):
            from google.colab import drive

            drive.mount("/content/drive")

        root_dir = "/content/drive/MyDrive/"
        work_dir = root_dir + "MusicGenerator/"
        batch_size = 256
    else:
        root_dir = "./"
        work_dir = root_dir
        batch_size = 3

    return Config(root_dir=root_dir, work_dir=work_dir, batch_size=batch_size)


config = create_config()
sys.path.append(config.root_dir)
sys.path.append(config.work_dir)
import my_common as my
import audio_common


logger = my.create_logger()


class EfficientAttention(nn.Module):
    def __init__(self, n_embd=512, n_head=8):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # 512 // 8 = 64 channels per head

        # Combined projections for all 8 heads into single, highly optimized matrix operations
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.key_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.val_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Final output projection layer
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Batch size , T: Block size, C: Embedding dimension
        B, T, C = x.shape

        # Project inputs and reshape to isolate the attention heads
        # Tensor transformation flow: [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.val_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Execute PyTorch's native Scaled Dot-Product Attention (SDPA)
        # This triggers FlashAttention kernels under the hood.
        # It is mathematically identical to old loop but prevents the massive VRAM overhead.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=config.dropout if self.training else 0.0
        )

        # Concatenate all attention heads back into a single feature tensor
        # Tensor transformation flow: [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Apply final linear projection mapping
        out = self.proj(out)
        # And a final dropout
        out = self.dropout(out)
        return out


class FeedForwad(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = EfficientAttention(n_embd, n_head)
        self.ffwd = FeedForwad(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class EnCodecEmbedding(nn.Module):
    def __init__(self, codebook_size, dim):
        super().__init__()
        self.token_emb = nn.Embedding(codebook_size, dim)

    def forward(self, codes):
        # codes: (B,T)
        return self.token_emb(codes)


class TransformerAudioModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = EnCodecEmbedding(
            codebook_size=vocab_size,
            dim=config.n_embed,
        )
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(config.n_embed, n_head=config.n_head)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embed)  # final layer norm
        self.lm_head = nn.Linear(config.n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)
        # B (Batch size) T (Time / Tokens) C (Channels / Embedding size)
        B, T, C = tok_emb.shape

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=my.DEVICE)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B_l, T_l, C_l = logits.shape

            logits = logits.reshape(B_l * T_l, C_l)
            targets = targets.reshape(B_l * T_l)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, codes, max_new_tokens, bos_token):
        B, T = codes.shape

        idx = codes
        for _ in range(max_new_tokens):

            if idx.shape[1] > config.block_size:
                idx_cond = idx[:, -config.block_size :]
            else:
                idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            # Prevent BOS from being generated after the start token.
            logits[:, bos_token] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class TransformerAudioTrainer:
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
            num_workers=2,
            persistent_workers=True,  # Prevents RAM leaks across epochs
        )

        self.checkpoint_manager = audio_common.CheckpointManager(
            model_file=config.model_file, lr=config.lr, device=my.DEVICE, logger=logger
        )

        self.vocab_size = self.encodec.config.codebook_size

        # Add a special BOS (beginning-of-sequence) token for starting generation.
        # Increase model vocabulary size to include BOS. Sequences start as:
        # [BOS, 34, 45, ...]
        self.bos_token = self.vocab_size
        self.model_vocab_size = self.vocab_size + 1

        self.num_codebooks = self.encodec.config.num_quantizers

        # Instantiate the generative transformer model
        self.model = TransformerAudioModel(vocab_size=self.model_vocab_size).to(
            my.DEVICE
        )
        self.evaluator = audio_common.AudioFidelityEvaluator(
            self.encodec, sample_rate=self.sr, device=my.DEVICE
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scaler = torch.cuda.amp.GradScaler()

        self.model, self.optimizer = self.checkpoint_manager.load_model(
            self.model, self.optimizer
        )

        my.print_parameter_summary(self.model)
        self.evaluator = audio_common.AudioFidelityEvaluator(
            self.encodec, sample_rate=self.sr, device=my.DEVICE
        )

        self.loss_history = deque(maxlen=1000)
        self.visualizer = audio_common.AudioVisualizer(logger=logger)

        self.start_epoch = 0

    def update_avg_loss(self, loss_value):
        self.loss_history.append(loss_value)
        return sum(self.loss_history) / len(self.loss_history)

    def train(self):
        step = 0  # Global step counter across epochs
        self.model.train()
        logger.info(
            f"Training started!\n"
            f"Number of data samples: {len(self.dataset)}\n"
            f"Batch Size: {config.batch_size}\n"
            f"Batches per epoch: {len(self.loader)}\n"
            f"LR: {config.lr}"
        )

        for epoch in range(self.start_epoch, config.epochs):

            for batch_idx, batch in enumerate(self.loader):
                # Extract raw continuous audio from the EnCodec processor/collator output
                waveform = batch["input_values"].to(my.DEVICE)
                # Extract discrete acoustic tokens (codes) using the EnCodec Encoder
                with torch.no_grad():
                    encoder_outputs = self.encodec.encode(waveform)
                    codes = encoder_outputs.audio_codes
                    # (nb_frames,B,n_q,T) -> (B,n_q,T)
                    codes = codes.squeeze(0)
                    # keep codebook 0
                    codes = codes[:, 0, :]
                    B, T = codes.shape
                    # Start sequence with BOS token.
                    bos_tokens = torch.full(
                        (B, 1),
                        self.bos_token,
                        dtype=torch.long,
                        device=my.DEVICE,
                    )
                    idx = torch.cat([bos_tokens, codes[:, :-1]], dim=1)
                    targets = codes

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    logits, loss = self.model(idx, targets)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer update
                self.scaler.step(self.optimizer)
                self.scaler.update()

                avg_loss = self.update_avg_loss(loss.item())
                step += 1  # Increment the global step counter

                # Periodic evaluation, saving, logging, and visualization every 20 steps
                if step % 20 == 0:
                    self.model.eval()
                    self.checkpoint_manager.save_checkpoint(self.model, self.optimizer)

                    # Match your exact requested logging template
                    logger.info(
                        f"Epoch {epoch} "
                        f"Step {step} "
                        f"Loss: {loss.item():.4f} "
                        f"Avg Loss: {avg_loss:.4f}"
                    )

                    # Generate an audio sample from scratch using the BOS token as the initial context.
                    # The model learns to start generation after the BOS (beginning-of-sequence) token.
                    prime_tokens = torch.tensor([[self.bos_token]], device=my.DEVICE)

                    generated_wave = self.generate_audio(
                        prime_tokens, max_new_tokens=200
                    )
                    generated_sample = generated_wave.cpu().squeeze().numpy()

                    # Use visualizer and state logger in the exact requested sequence
                    self.visualizer.show_samples(
                        generated_sample, self.dataset, self.sr
                    )

                    self.model.train()

            logger.info(f"Epoch {epoch} completed.")

    @torch.no_grad()
    def generate_audio(self, prime_tokens, max_new_tokens):
        self.model.eval()

        generated_cb0 = self.model.generate(
            prime_tokens.to(my.DEVICE), max_new_tokens, self.bos_token
        )

        # Remove BOS token because EnCodec decoder only understands audio tokens.
        generated_cb0 = generated_cb0[:, 1:]
        B, T = generated_cb0.shape

        # Recreate EnCodec shape. Since we only generate CB0, fill the remaining
        # codebooks with zeros for decoding.
        generated_codes = torch.zeros(
            1, B, self.num_codebooks, T, dtype=torch.long, device=my.DEVICE
        )

        # put generated tokens into codebook 0
        generated_codes[:, :, 0, :] = generated_cb0
        audio_output = self.encodec.decode(generated_codes, [None], None)

        return audio_output.audio_values


def train():
    audio_common.ensure_dir(config.out_dir, logger=logger)
    trainer = TransformerAudioTrainer()
    trainer.train()


if __name__ == "__main__":
    train()
