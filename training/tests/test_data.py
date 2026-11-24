from pathlib import Path
from PIL import Image
import numpy as np
import pytest
from weatherdiff.data import CDD11Dataset


def _png(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8), 'RGB').save(path)


def test_corrupt_shared_gt_excludes_all_corresponding_pairs(tmp_path):
    root = tmp_path / 'cdd'
    (root / 'splits').mkdir(parents=True)
    (root / 'splits/train.txt').write_text('good.png\nbad.png\n')
    (root / 'splits/val.txt').write_text('good.png\n')
    for cat in ('clear', 'haze', 'rain'):
        _png(root / 'train' / cat / 'good.png', 100)
        if cat == 'clear':
            path = root / 'train' / cat / 'bad.png'
            path.write_bytes(b'not an image')
        else:
            _png(root / 'train' / cat / 'bad.png', 50)
    ds = CDD11Dataset(root, 'train', crop_size=8, index_dir=tmp_path / 'index')
    assert len(ds) == 2
    assert all(name == 'good.png' for _, name in ds.samples)
    assert ds[0]['input'].shape == (3, 8, 8)
    ds2 = CDD11Dataset(root, 'train', crop_size=8, index_dir=tmp_path / 'index')
    assert ds.samples == ds2.samples
