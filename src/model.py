"""Music generation models.

MusicVAE  – convolutional VAE that encodes/decodes mel-spectrogram chunks.
LatentLSTM – autoregressive LSTM that models sequences of VAE latent codes,
             giving temporal coherence across chunks at generation time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, kernel: int = 4,
                stride: int = 2, pad: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int, kernel: int = 4,
                  stride: int = 2, pad: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Convolutional VAE
# ---------------------------------------------------------------------------

class MusicVAE(nn.Module):
    """Convolutional VAE for mel-spectrogram chunks.

    Input : (B, 1, N_MELS, FRAMES_PER_CHUNK) normalised to [-1, 1].
    Latent: (B, latent_dim).
    """

    def __init__(self, n_mels: int = 128, n_frames: int = 172, latent_dim: int = 128):
        super().__init__()
        self.n_mels    = n_mels
        self.n_frames  = n_frames
        self.latent_dim = latent_dim

        # ---- Encoder -------------------------------------------------------
        self.enc_conv = nn.Sequential(
            _conv_block(1,    32),   # → (B, 32,  64, 86)
            _conv_block(32,   64),   # → (B, 64,  32, 43)
            _conv_block(64,  128),   # → (B, 128, 16, 21)
            _conv_block(128, 256),   # → (B, 256,  8, 10)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            enc_out = self.enc_conv(dummy)
        self._enc_spatial = tuple(enc_out.shape[1:])   # (256, h, w)
        flat = int(np.prod(self._enc_spatial))

        self.enc_mu     = nn.Linear(flat, latent_dim)
        self.enc_logvar = nn.Linear(flat, latent_dim)

        # ---- Decoder -------------------------------------------------------
        self.dec_fc = nn.Linear(latent_dim, flat)

        self.dec_conv = nn.Sequential(
            _deconv_block(256, 128),
            _deconv_block(128,  64),
            _deconv_block(64,   32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

    # ---- API ---------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, *self._enc_spatial)
        out = self.dec_conv(h)
        if out.shape[2:] != (self.n_mels, self.n_frames):
            out = F.interpolate(
                out, size=(self.n_mels, self.n_frames),
                mode="bilinear", align_corners=False,
            )
        return out

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss(
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # MSE on normalised mel values
        mse = F.mse_loss(recon, target)
        # Log-magnitude spectral loss: map [-1,1]→(0,2] then take log
        # This penalises errors more strongly in quiet regions, matching
        # how human hearing perceives loudness logarithmically.
        r_log = torch.log((recon + 1.0).clamp(min=1e-6))
        t_log = torch.log((target + 1.0).clamp(min=1e-6))
        log_loss  = F.l1_loss(r_log, t_log)
        recon_loss = mse + 0.3 * log_loss
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Latent LSTM Prior
# ---------------------------------------------------------------------------

class LatentLSTM(nn.Module):
    """Autoregressive LSTM that predicts the next VAE latent code.

    Trained on sequences of consecutive codes extracted from real audio.
    At generation time, autoregressively produces temporally coherent
    latent sequences which are decoded by MusicVAE.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lstm = nn.LSTM(
            latent_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, latent_dim) → predictions: (B, T, latent_dim)"""
        out, _ = self.lstm(x)
        return self.proj(out)

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    def generate(
        self,
        n_steps: int,
        device: str = "cuda",
        temperature: float = 1.0,
        seed_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate n_steps latent codes.
        Returns tensor of shape (n_steps, latent_dim).
        """
        self.eval()
        with torch.no_grad():
            dev = torch.device(device)
            if seed_z is not None:
                z = seed_z.to(dev).unsqueeze(0).unsqueeze(0)   # (1, 1, latent_dim)
            else:
                z = torch.randn(1, 1, self.latent_dim, device=dev) * temperature
            hidden: tuple | None = None
            codes: list[torch.Tensor] = []

            for _ in range(n_steps):
                lstm_out, hidden = self.lstm(z, hidden)
                pred = self.proj(lstm_out)
                # Small diversity noise; kept tiny to prevent latent drift across many steps.
                # 0.02 * temperature: at temp=1.0 this is ~2% of the typical latent magnitude.
                pred = pred + torch.randn_like(pred) * 0.02 * temperature
                codes.append(pred.squeeze(0).squeeze(0))   # (latent_dim,)
                z = pred

        return torch.stack(codes)   # (n_steps, latent_dim)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
