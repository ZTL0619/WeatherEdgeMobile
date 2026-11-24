"""Standard full-image RGB PSNR and Gaussian-window SSIM, per image, on [0,1]."""
import torch
import torch.nn.functional as F


def psnr_batch(x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = (x.clamp(0, 1) - gt).square().flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-12))


def ssim_batch(x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if x.shape != gt.shape or x.shape[1] != 3:
        raise ValueError('SSIM expects matching [B,3,H,W] tensors')
    x = x.clamp(0, 1).float()
    gt = gt.float()
    size = min(11, x.shape[-2], x.shape[-1])
    if size % 2 == 0:
        size -= 1
    if size < 3:
        raise ValueError('SSIM input too small')
    axis = torch.arange(size, device=x.device, dtype=x.dtype) - size // 2
    gauss = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    kernel = torch.outer(gauss, gauss)[None, None].expand(3, 1, -1, -1)
    conv = lambda z: F.conv2d(z, kernel, groups=3)
    mx, my = conv(x), conv(gt)
    vx = (conv(x * x) - mx.square()).clamp_min(0)
    vy = (conv(gt * gt) - my.square()).clamp_min(0)
    vxy = conv(x * gt) - mx * my
    c1, c2 = .01 ** 2, .03 ** 2
    ssim = ((2 * mx * my + c1) * (2 * vxy + c2) /
            ((mx.square() + my.square() + c1) * (vx + vy + c2)))
    return ssim.flatten(1).mean(1)


def gradient_loss(x, y):
    return ((x[:, :, 1:] - x[:, :, :-1] - y[:, :, 1:] + y[:, :, :-1]).abs().mean() +
            (x[:, :, :, 1:] - x[:, :, :, :-1] - y[:, :, :, 1:] + y[:, :, :, :-1]).abs().mean()) * .5
