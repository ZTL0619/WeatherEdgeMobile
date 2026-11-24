"""CDD11 pair indexing, corrupt-image filtering and aligned patches.

Structure: root/train/{clear,haze,...}/NAME.png; root/test/{clear,...};
root/splits/{train,val}.txt.  No files are modified under root.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

DEFAULT_DATA_ROOT = os.environ.get('CDD11_ROOT', 'data/CDD11')


def _verify_png(path: Path):
    try:
        with Image.open(path) as im:
            im.load()  # fully decode rather than only checking header
            return im.size
    except (OSError, ValueError, SyntaxError) as exc:
        return f'{type(exc).__name__}: {exc}'


def _load_png(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert('RGB')


def _tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(im, dtype=np.uint8).transpose(2, 0, 1).copy()).float().div_(255.)


def _key(category: str, name: str, seed: int) -> int:
    data = f'{category}/{name}|{seed}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), byteorder='little')


class CDD11Dataset(Dataset):
    def __init__(self, root: str | Path = DEFAULT_DATA_ROOT, split: str = 'train',
                 crop_size: int | None = 256, *, index_dir: str | Path = 'outputs/index',
                 verify: bool = True, fixed_crop: bool = False, seed: int = 42,
                 augment: bool = False):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.crop_size = crop_size if split == 'train' else None
        self.fixed_crop = fixed_crop
        self.seed = seed
        self.augment = augment
        self.folder = self.root / ('test' if split == 'test' else 'train')
        if not self.folder.is_dir() or not (self.folder / 'clear').is_dir():
            raise FileNotFoundError(f'CDD11 data folder missing: {self.folder}')
        categories = sorted(p.name for p in self.folder.iterdir() if p.is_dir() and p.name != 'clear')
        if not categories:
            raise RuntimeError(f'No degradation categories found in {self.folder}')
        self.categories = categories
        if split == 'test':
            names = sorted(p.name for p in (self.folder / 'clear').glob('*.png'))
        else:
            split_file = self.root / 'splits' / f'{split}.txt'
            if not split_file.is_file():
                raise FileNotFoundError(split_file)
            names = [n.strip() for n in split_file.read_text().splitlines() if n.strip()]
        candidates = [(cat, name) for name in names for cat in categories]
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        manifest = index_dir / f'cdd11_{split}.json'
        expected = {'root': str(self.root), 'split': split, 'crop_size': self.crop_size,
                    'categories': categories, 'names_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest()}
        cached = None
        if manifest.exists() and verify:
            try:
                payload = json.loads(manifest.read_text())
                if payload['identity'] == expected:
                    cached = [tuple(x) for x in payload['valid']]
                    print(f'[CDD11 {split}] using verified index: {manifest} ({len(cached)} pairs)', flush=True)
            except (OSError, ValueError, KeyError):
                cached = None
        if cached is not None:
            self.samples = cached
        elif verify:
            self.samples = []
            bad = []
            gt_status = {}
            for i, (cat, name) in enumerate(candidates):
                gt = self.folder / 'clear' / name
                inp = self.folder / cat / name
                if name not in gt_status:
                    gt_status[name] = _verify_png(gt)
                target_size = gt_status[name]
                input_size = _verify_png(inp) if isinstance(target_size, tuple) else 'GT invalid'
                reason = None
                if not isinstance(target_size, tuple):
                    reason = f'GT {gt}: {target_size}'
                elif not isinstance(input_size, tuple):
                    reason = f'input {inp}: {input_size}'
                elif target_size != input_size:
                    reason = f'size mismatch input {input_size} vs GT {target_size}'
                elif self.crop_size and min(target_size) < self.crop_size:
                    reason = f'images smaller than crop size {self.crop_size}: {target_size}'
                if reason:
                    bad.append((cat, name, reason))
                else:
                    self.samples.append((cat, name))
                if (i + 1) % 1000 == 0:
                    print(f'[CDD11 {split}] verified {i + 1}/{len(candidates)}, skipped {len(bad)}', flush=True)
            temp = manifest.with_suffix('.json.tmp')
            temp.write_text(json.dumps({'identity': expected, 'valid': self.samples}, indent=2))
            os.replace(temp, manifest)
            report = index_dir / f'broken_{split}.csv'
            with report.open('w', newline='') as file:
                w = csv.writer(file)
                w.writerow(['category', 'filename', 'reason'])
                w.writerows(bad)
            print(f'[CDD11 {split}] verified {len(candidates)}, kept {len(self.samples)}, skipped {len(bad)}; report={report}', flush=True)
        else:
            self.samples = candidates
        if not self.samples:
            raise RuntimeError(f'No valid samples in CDD11 {split}')

    def __len__(self):
        return len(self.samples)

    def paths(self, index):
        cat, name = self.samples[index]
        return self.folder / cat / name, self.folder / 'clear' / name

    def crop_box(self, cat: str, name: str, w: int, h: int):
        assert self.crop_size is not None
        s = self.crop_size
        if min(w, h) < s:
            raise RuntimeError(f'image smaller than crop size: {cat}/{name}, {w}x{h} < {s}')
        rng = random.Random(_key(cat, name, self.seed)) if self.fixed_crop else random
        left, top = rng.randint(0, w - s), rng.randint(0, h - s)
        return (left, top, left + s, top + s)

    def __getitem__(self, index):
        cat, name = self.samples[index]
        ip, gp = self.paths(index)
        y, gt = _load_png(ip), _load_png(gp)
        if y.size != gt.size:
            raise RuntimeError(f'image dimension mismatch {ip} / {gp}')
        if self.crop_size is not None:
            box = self.crop_box(cat, name, *y.size)
            y, gt = y.crop(box), gt.crop(box)
        if self.augment and self.split == 'train':
            if random.random() < .5:
                y, gt = ImageOps.mirror(y), ImageOps.mirror(gt)
            if random.random() < .5:
                y, gt = ImageOps.flip(y), ImageOps.flip(gt)
        return {'input': _tensor(y), 'target': _tensor(gt),
                'category': cat, 'filename': name}


class DistillationDataset(Dataset):
    """Fixed deterministic 256px crop, paired teacher-cache PNG, synchronized flips."""
    def __init__(self, dataset: CDD11Dataset, cache_dir: str | Path, augment=True):
        if dataset.split != 'train' or not dataset.fixed_crop or dataset.crop_size is None:
            raise ValueError('Distillation requires train dataset with fixed_crop=True')
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.augment = augment
        missing = [f'{cat}/{name}' for cat, name in dataset.samples if not (self.cache_dir / cat / name).is_file()]
        if missing:
            raise RuntimeError(f'{len(missing)} teacher pseudo-targets missing; example: {missing[0]}; run cache-teacher')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cat, name = self.dataset.samples[idx]
        ip, gp = self.dataset.paths(idx)
        y, gt = _load_png(ip), _load_png(gp)
        pseudo = _load_png(self.cache_dir / cat / name)
        box = self.dataset.crop_box(cat, name, *y.size)
        y, gt = y.crop(box), gt.crop(box)
        if pseudo.size != y.size:
            raise RuntimeError(f'cached teacher prediction size mismatch for {cat}/{name}')
        if self.augment:
            if random.random() < .5:
                y, gt, pseudo = map(ImageOps.mirror, (y, gt, pseudo))
            if random.random() < .5:
                y, gt, pseudo = map(ImageOps.flip, (y, gt, pseudo))
        return {'input': _tensor(y), 'target': _tensor(gt), 'teacher': _tensor(pseudo),
                'category': cat, 'filename': name}
