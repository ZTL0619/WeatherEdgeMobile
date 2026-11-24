"""Deterministic MobileIE-inspired image restoration model.

Adapted ideas, NOT an exact reproduction of MobileIE (ICCV 2025):
- MBRConv-style algebraically deployable multi-branch convolutions
- feature self-transform and lightweight dual-path attention
- compact 3-scale encoder/decoder
No depthwise conv, Gaussian noise, timestep or diffusion sampler in student.
"""
from __future__ import annotations

import copy
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size not in (3, 5):
            raise ValueError("kernel_size must be 3 or 5")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        k = kernel_size
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=shape,
                          padding=(shape[0] // 2, shape[1] // 2), bias=False),
                nn.BatchNorm2d(out_ch),
            )
            for shape in ((k, k), (1, k), (k, 1), (1, 1))
        ])
        self.mix = nn.Conv2d(4 * out_ch, out_ch, 1)
        self.fused = None

    def forward(self, x: Tensor) -> Tensor:
        if self.fused is not None:
            return self.fused(x)
        return self.mix(torch.cat([b(x) for b in self.branches], dim=1))

    @torch.no_grad()
    def fuse(self) -> None:
        if self.fused is not None:
            return
        if self.training:
            raise RuntimeError("Switch the model to eval() before fusing BN")
        ref = self.mix.weight
        k = self.kernel_size
        merged = nn.Conv2d(self.in_ch, self.out_ch, k, padding=k // 2,
                           bias=True).to(device=ref.device, dtype=ref.dtype)
        weight = torch.zeros_like(merged.weight)
        bias = self.mix.bias.detach().clone()
        for i, branch in enumerate(self.branches):
            conv, bn = branch
            factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = conv.weight * factor[:, None, None, None]
            b = bn.bias - bn.running_mean * factor
            pt = (k - w.shape[-2]) // 2
            pl = (k - w.shape[-1]) // 2
            w = F.pad(w, (pl, k - w.shape[-1] - pl, pt, k - w.shape[-2] - pt))
            mix = self.mix.weight[:, i * self.out_ch:(i + 1) * self.out_ch, 0, 0]
            weight.add_(torch.einsum('oc,cihw->oihw', mix, w))
            bias.add_(torch.mv(mix, b))
        merged.weight.copy_(weight)
        merged.bias.copy_(bias)
        self.fused = merged
        del self.branches
        del self.mix


class FST(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, ch, 1, 1), 0.001))
        self.bias = nn.Parameter(torch.zeros((1, ch, 1, 1)))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * x.square() + self.bias


class HDPA(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.global_gate = nn.Conv2d(ch, ch, 1)
        self.local_gate = nn.Conv2d(ch, ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        global_attention = torch.sigmoid(self.global_gate(x.mean(dim=(2, 3), keepdim=True)))
        local_attention = torch.sigmoid(self.local_gate(F.max_pool2d(
            x * global_attention, kernel_size=3, stride=1, padding=1)))
        return x * (0.5 + global_attention * local_attention)


class RestoreBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = RepConv(ch, ch)
        self.act = nn.PReLU(ch)
        self.fst = FST(ch)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.25 * self.fst(self.act(self.conv(x)))


class MobileIEStudent(nn.Module):
    """Only argument: degraded RGB image in [0,1]. Returns enhanced RGB.

    Deployment model is a deterministic f_theta(y), with one forward pass.
    """
    def __init__(self, width: int = 32):
        super().__init__()
        if width < 16 or width % 4:
            raise ValueError("width must be >=16 and divisible by 4")
        self.width = width
        c0, c1, c2 = width // 2, width, width * 3 // 2
        self.stem = nn.Sequential(nn.Conv2d(3, c0, 3, padding=1), nn.PReLU(c0))
        self.down1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(RestoreBlock(c1), RestoreBlock(c1))
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.mid = nn.Sequential(RestoreBlock(c2), RestoreBlock(c2), RestoreBlock(c2))
        self.attn = HDPA(c2)
        self.merge1 = nn.Conv2d(c2 + c1, c1, 1)
        self.dec1 = RestoreBlock(c1)
        self.merge0 = nn.Conv2d(c1 + c0, c0, 1)
        self.dec0 = RestoreBlock(c0)
        self.out = nn.Conv2d(c0, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, degraded: Tensor) -> Tensor:
        if degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError("Expected degraded RGB tensor [B,3,H,W]")
        height, width = degraded.shape[-2:]
        dh, dw = (-height) % 4, (-width) % 4
        y = F.pad(degraded, (0, dw, 0, dh), mode='replicate') if dh or dw else degraded
        s0 = self.stem(y)
        s1 = self.enc1(self.down1(s0))
        z = self.attn(self.mid(self.down2(s1)))
        d1 = self.dec1(self.merge1(torch.cat((F.interpolate(z, size=s1.shape[-2:], mode='nearest'), s1), dim=1)))
        d0 = self.dec0(self.merge0(torch.cat((F.interpolate(d1, size=s0.shape[-2:], mode='nearest'), s0), dim=1)))
        return (y + self.out(d0))[..., :height, :width]

    def fused_copy(self):
        fused = copy.deepcopy(self).eval()
        for layer in fused.modules():
            if isinstance(layer, RepConv):
                layer.fuse()
        return fused
