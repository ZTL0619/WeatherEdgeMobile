from __future__ import annotations
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import DEFAULT_DATA_ROOT, CDD11Dataset
from .diffusion import ConditionalDiffusionUNet, GaussianDiffusion
from .student import MobileIEStudent


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_args(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, torch.cuda.is_available() is false. Check uv/PyTorch CUDA installation.')
    return device


def loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=shuffle, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def safe_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def history_append(path, row):
    path = Path(path)
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def rgb_to_image(rgb_tensor):
    arr = (rgb_tensor.detach().clamp(0, 1).mul(255).add(.5).byte()
           .permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr, 'RGB')


def image_to_tensor(image):
    return torch.from_numpy(np.array(image.convert('RGB'), dtype=np.uint8).transpose(2, 0, 1).copy()).float() / 255.


def load_teacher(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    c = ck['config']
    if c.get('architecture') != 'conditional_ddpm_x0_v1':
        raise ValueError(f"Not a compatible diffusion-teacher checkpoint: {c.get('architecture')}")
    model = ConditionalDiffusionUNet(c['base']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    schedule = GaussianDiffusion(c['timesteps']).to(device)
    return model, schedule, c


def load_student(checkpoint, device, fuse=True):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    c = ck['config']
    if c.get('architecture') != 'mobileie_deterministic_v1':
        raise ValueError(f"Not a compatible deterministic student: {c.get('architecture')}")
    model = MobileIEStudent(c['width']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return (model.fused_copy() if fuse else model), c


@torch.inference_mode()
def evaluate_loader(model_type, model, schedule, eval_loader, device, *, teacher_steps=20,
                    limit=0, seed=42, ssim=False):
    """Accuracy aggregated per sample, not per batch; stable noise for teacher."""
    from .metrics import psnr_batch, ssim_batch
    totals = {}
    n = 0
    for batch in eval_loader:
        y = batch['input'].to(device)
        gt = batch['target'].to(device)
        if model_type == 'teacher':
            g = torch.Generator(device=device).manual_seed(seed + n)
            noise = torch.randn(y.shape, generator=g, device=device, dtype=y.dtype)
            out = schedule.sample(model, y, steps=teacher_steps, noise=noise)
        else:
            out = model(y).clamp(0, 1)
        ps = psnr_batch(out, gt).cpu().tolist()
        ss = ssim_batch(out, gt).cpu().tolist() if ssim else [0.] * len(ps)
        ip = psnr_batch(y, gt).cpu().tolist()
        for j, cat in enumerate(batch['category']):
            cell = totals.setdefault(cat, [0., 0., 0., 0])
            cell[0] += ps[j]
            cell[1] += ss[j]
            cell[2] += ip[j]
            cell[3] += 1
        n += len(ps)
        if limit and n >= limit:
            break
    if not n:
        raise RuntimeError('Empty evaluation set')
    rows = {}
    for cat, (ps, ss, inp, count) in sorted(totals.items()):
        rows[cat] = {'category': cat, 'count': count, 'input_psnr': inp / count,
                     'psnr': ps / count, 'ssim': ss / count if ssim else float('nan')}
    count = sum(row['count'] for row in rows.values())
    rows['ALL'] = {'category': 'ALL', 'count': count,
                   'input_psnr': sum(row['input_psnr'] * row['count'] for row in rows.values()) / count,
                   'psnr': sum(row['psnr'] * row['count'] for row in rows.values()) / count,
                   'ssim': (sum(row['ssim'] * row['count'] for row in rows.values()) / count
                            if ssim else float('nan'))}
    return rows


def write_eval_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=('category', 'count', 'input_psnr', 'psnr', 'ssim'))
        w.writeheader()
        w.writerows(rows.values())
