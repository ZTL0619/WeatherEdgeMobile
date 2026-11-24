"""Actual conditional denoising diffusion teacher: q(x_t | x0) is defined on CLEAN GT.

Training: draw t ~ Uniform[0,T-1], epsilon ~ Normal(0,I), predict x0 from
x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon, conditioned on y.
Sampling: deterministic DDIM (eta=0), starting at x_T~Normal(0,I), with
multiple calls to the teacher.  No Gaussian noise is ever added to the degraded
condition y.  This teacher is not the legacy deterministic Fusion Teacher.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device,
                       dtype=torch.float32) / max(1, half - 1))
    phase = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([phase.sin(), phase.cos()], 1)
    return F.pad(emb, (0, dim - emb.shape[-1]))


class TimeResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int):
        super().__init__()
        self.block1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.time_proj = nn.Linear(time_ch, out_ch)
        self.block2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x) + self.time_proj(F.silu(emb))[:, :, None, None]
        return (self.skip(x) + self.block2(h)) * (2.0 ** -0.5)


class ConditionalDiffusionUNet(nn.Module):
    """Conditional 3-scale U-Net, input (x_t, degraded y, t); output x0 estimate."""
    def __init__(self, base: int = 48):
        super().__init__()
        if base < 16 or base % 8:
            raise ValueError('Teacher base must be >=16 and divisible by 8')
        self.base = base
        a, b, c, d = base, base * 2, base * 3, base * 4
        tc = base * 4
        self.time = nn.Sequential(nn.Linear(tc, tc), nn.SiLU(), nn.Linear(tc, tc))
        self.input = nn.Conv2d(6, a, 3, padding=1)
        self.e0 = nn.ModuleList([TimeResBlock(a, a, tc), TimeResBlock(a, a, tc)])
        self.down0 = nn.Conv2d(a, b, 3, stride=2, padding=1)
        self.e1 = nn.ModuleList([TimeResBlock(b, b, tc), TimeResBlock(b, b, tc)])
        self.down1 = nn.Conv2d(b, c, 3, stride=2, padding=1)
        self.e2 = nn.ModuleList([TimeResBlock(c, c, tc), TimeResBlock(c, c, tc)])
        self.down2 = nn.Conv2d(c, d, 3, stride=2, padding=1)
        self.middle = nn.ModuleList([TimeResBlock(d, d, tc), TimeResBlock(d, d, tc)])
        self.merge2 = nn.Conv2d(d + c, c, 1)
        self.d2 = nn.ModuleList([TimeResBlock(c, c, tc), TimeResBlock(c, c, tc)])
        self.merge1 = nn.Conv2d(c + b, b, 1)
        self.d1 = nn.ModuleList([TimeResBlock(b, b, tc), TimeResBlock(b, b, tc)])
        self.merge0 = nn.Conv2d(b + a, a, 1)
        self.d0 = nn.ModuleList([TimeResBlock(a, a, tc), TimeResBlock(a, a, tc)])
        self.output = nn.Sequential(nn.GroupNorm(8, a), nn.SiLU(), nn.Conv2d(a, 3, 3, padding=1))
        # A good initial conditional estimate is the degraded image itself.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _forward_blocks(blocks, x, emb):
        for block in blocks:
            x = block(x, emb)
        return x

    def forward(self, xt: torch.Tensor, degraded: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if xt.shape != degraded.shape or xt.ndim != 4 or xt.shape[1] != 3:
            raise ValueError('xt and degraded must both be [B,3,H,W]')
        if t.ndim != 1 or t.shape[0] != xt.shape[0]:
            raise ValueError('t must be [B]')
        h, w = degraded.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            xt = F.pad(xt, (0, pw, 0, ph), mode='replicate')
            degraded = F.pad(degraded, (0, pw, 0, ph), mode='replicate')
        emb = self.time(sinusoidal_embedding(t, self.base * 4))
        h0 = self._forward_blocks(self.e0, self.input(torch.cat([xt, degraded], 1)), emb)
        h1 = self._forward_blocks(self.e1, self.down0(h0), emb)
        h2 = self._forward_blocks(self.e2, self.down1(h1), emb)
        z = self._forward_blocks(self.middle, self.down2(h2), emb)
        z = self._forward_blocks(self.d2, self.merge2(torch.cat([F.interpolate(z, size=h2.shape[-2:], mode='nearest'), h2], 1)), emb)
        z = self._forward_blocks(self.d1, self.merge1(torch.cat([F.interpolate(z, size=h1.shape[-2:], mode='nearest'), h1], 1)), emb)
        z = self._forward_blocks(self.d0, self.merge0(torch.cat([F.interpolate(z, size=h0.shape[-2:], mode='nearest'), h0], 1)), emb)
        # Input tensors [-1,1] training and sampling; y+residual estimate of x0.
        return (degraded + self.output(z))[..., :h, :w]


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000):
        super().__init__()
        if timesteps < 2:
            raise ValueError('Diffusion timesteps must be >=2')
        self.timesteps = int(timesteps)
        # Improved DDPM cosine cumulative variance schedule.
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        ac = torch.cos((x / timesteps + .008) / 1.008 * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(0.0001, 0.999)
        alpha_bar = (1 - betas).cumprod(0).float()
        self.register_buffer('alpha_bar', alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def sample(self, teacher: nn.Module, degraded: torch.Tensor, *, steps: int = 20,
               noise: torch.Tensor | None = None, clip_x0: bool = True) -> torch.Tensor:
        """DDIM eta=0 with exactly `steps` teacher forward calls.

        Image y is never perturbed. Inputs y in [0,1], output also in [0,1].
        """
        if not 2 <= steps <= self.timesteps:
            raise ValueError('Sampling steps must be in [2, train_timesteps]')
        if degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError('degraded must have shape [B,3,H,W]')
        y = degraded * 2 - 1
        xt = torch.randn_like(y) if noise is None else noise.clone()
        if xt.shape != y.shape:
            raise ValueError('noise and degraded image must have the same shape')
        indices = torch.linspace(self.timesteps - 1, 0, steps, device=y.device).round().long().tolist()
        for i, t in enumerate(indices):
            ts = torch.full((y.shape[0],), t, device=y.device, dtype=torch.long)
            x0 = teacher(xt, y, ts)
            if clip_x0:
                x0 = x0.clamp(-1, 1)
            if i == len(indices) - 1:
                xt = x0
                break
            t_next = indices[i + 1]
            ab = self.alpha_bar[t]
            ab_next = self.alpha_bar[t_next]
            epsilon = (xt - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-8)
            xt = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * epsilon
        return ((xt + 1) / 2).clamp(0, 1)
