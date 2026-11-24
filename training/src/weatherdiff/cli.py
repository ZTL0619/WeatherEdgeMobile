"""Command-line pipeline: teacher DDPM -> DDIM pseudo-labels -> deterministic student."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .common import (device_from_args, evaluate_loader, history_append, image_to_tensor,
                     load_student, load_teacher, loader, rgb_to_image, safe_save,
                     seed_everything, write_eval_csv)
from .data import CDD11Dataset, DEFAULT_DATA_ROOT, DistillationDataset
from .diffusion import ConditionalDiffusionUNet, GaussianDiffusion
from .metrics import gradient_loss, psnr_batch, ssim_batch
from .student import MobileIEStudent


def add_dataset_args(p):
    p.add_argument('--data-root', default=os.environ.get('CDD11_ROOT', DEFAULT_DATA_ROOT))
    p.add_argument('--index-dir', default='outputs/index', help='Persistent verified-file index')
    p.add_argument('--recheck-images', action='store_true', help='Remove old index files first to revalidate data')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)


def add_train_args(p):
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', default='', help='Resume from last.pt, not from best.pt')
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--val-limit', type=int, default=0, help='0=entire val set; >0 debug or faster subset only')
    p.add_argument('--max-train-batches', type=int, default=0, help='DEBUG ONLY; 0=all batches')


def get_dataset(args, split, crop=None, fixed_crop=False, augment=False):
    if getattr(args, 'recheck_images', False):
        path = Path(args.index_dir) / f'cdd11_{split}.json'
        path.unlink(missing_ok=True)
    return CDD11Dataset(args.data_root, split=split, crop_size=crop,
                        index_dir=args.index_dir, fixed_crop=fixed_crop,
                        augment=augment, seed=args.seed)


def _checkpoint_dict(config, model, optim, scheduler, scaler, epoch, best):
    return {'config': config, 'model': model.state_dict(), 'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'epoch': epoch, 'best_psnr': best}


def _resume_if_any(path, config, model, optim, scheduler, scaler):
    if not path:
        return 1, -float('inf')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['config'] != config:
        raise ValueError('Resume configuration mismatch; do not resume a different model or stage')
    model.load_state_dict(ck['model'])
    optim.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    scaler.load_state_dict(ck['scaler'])
    return ck['epoch'] + 1, ck['best_psnr']


def _train_checks(args):
    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 8 or args.val_every < 1:
        raise ValueError('Invalid epochs / batch-size / crop-size / val-every')
    if args.max_train_batches or args.val_limit:
        print('WARNING: subset/debug configuration enabled; metrics not full training/validation', flush=True)


def cmd_train_teacher(args):
    _train_checks(args)
    seed_everything(args.seed)
    device = device_from_args(args.device)
    amp = bool(args.amp and device.type == 'cuda')
    out = Path(args.outdir)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output folder not empty: {out}. Use --resume or new --outdir')
    out.mkdir(parents=True, exist_ok=True)
    train = get_dataset(args, 'train', crop=args.crop_size, augment=True)
    val = get_dataset(args, 'val', crop=None)
    tr_dl = loader(train, args.batch_size, args.workers, True)
    vl_dl = loader(val, 1, args.workers, False)
    model = ConditionalDiffusionUNet(args.base).to(device)
    diffusion = GaussianDiffusion(args.timesteps).to(device)
    config = {'architecture': 'conditional_ddpm_x0_v1', 'base': args.base, 'timesteps': args.timesteps}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best = _resume_if_any(args.resume, config, model, opt, sch, scaler)
    print(f'TEACHER: base={args.base}, params={sum(p.numel() for p in model.parameters()):,}, '
          f'diffusion steps T={args.timesteps}, DDIM validation steps={args.sample_steps}, '
          f'train={len(train)}, val={len(val)}', flush=True)
    for epoch in range(start, args.epochs + 1):
        model.train()
        summ = 0.
        count = 0
        tstart = time.perf_counter()
        bar = tqdm(tr_dl, desc=f'Teacher epoch {epoch}/{args.epochs}')
        for j, batch in enumerate(bar, 1):
            y = batch['input'].to(device, non_blocking=True).mul(2).sub(1)
            x0 = batch['target'].to(device, non_blocking=True).mul(2).sub(1)
            t = torch.randint(0, args.timesteps, (y.shape[0],), device=device)
            # IMPORTANT: only clean x0 is noised; degraded condition y is unchanged.
            xt = diffusion.q_sample(x0, t, torch.randn_like(x0))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=amp):
                prediction = model(xt, y, t)
                loss = F.l1_loss(prediction, x0) + .1 * F.mse_loss(prediction, x0)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt)
            scaler.update()
            summ += float(loss.detach())
            count += 1
            bar.set_postfix(loss=f'{summ/count:.4f}')
            if args.max_train_batches and j >= args.max_train_batches:
                break
        sch.step()
        measure = (epoch % args.val_every == 0) or (epoch == args.epochs)
        val_psnr = float('nan')
        if measure:
            model.eval()
            rows = evaluate_loader('teacher', model, diffusion, vl_dl, device,
                                   teacher_steps=args.sample_steps, limit=args.val_limit,
                                   seed=args.seed + 10000)
            val_psnr = rows['ALL']['psnr']
            print(f'TEACHER epoch={epoch} val PSNR={val_psnr:.3f} dB '
                  f'({rows["ALL"]["count"]} pairs)', flush=True)
        improved = measure and val_psnr > best
        if improved:
            best = val_psnr
        state = _checkpoint_dict(config, model, opt, sch, scaler, epoch, best)
        safe_save(out / 'last.pt', state)
        if improved:
            safe_save(out / 'best.pt', state)
            write_eval_csv(out / 'best_val_summary.csv', rows)
        history_append(out / 'history.csv', {'epoch': epoch, 'train_loss': summ / max(1, count),
                       'val_psnr': val_psnr, 'best_val_psnr': best,
                       'seconds': round(time.perf_counter() - tstart, 2)})
        print(f'TEACHER {epoch}/{args.epochs} loss={summ/max(1,count):.5f} '
              f'best={best:.3f} dB', flush=True)
    print(f'Teacher checkpoint: {out / "best.pt"}', flush=True)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _cache_identity(args):
    return {'teacher_sha256': _file_sha256(args.teacher), 'sample_steps': args.sample_steps,
            'seed': args.seed, 'crop_size': args.crop_size,
            'data_root': str(Path(args.data_root).expanduser().resolve())}


def cmd_cache_teacher(args):
    seed_everything(args.seed)
    device = device_from_args(args.device)
    train = get_dataset(args, 'train', crop=args.crop_size, fixed_crop=True)
    teacher, diffusion, _ = load_teacher(args.teacher, device)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    meta_file = cache / 'metadata.json'
    identity = _cache_identity(args)
    if meta_file.is_file():
        if json.loads(meta_file.read_text()) != identity:
            raise RuntimeError('Cache generated for another teacher/seed/steps/data. Use a NEW --cache-dir.')
    else:
        if any(cache.rglob('*.png')):
            raise RuntimeError('Cache has PNG files but no metadata; choose a new --cache-dir')
        meta_file.write_text(json.dumps(identity, indent=2))
    done = 0
    skipped = 0
    for i, (cat, name) in enumerate(tqdm(train.samples, desc='Teacher DDIM pseudo-targets')):
        dest = cache / cat / name
        if dest.is_file():
            skipped += 1
            continue
        example = train[i]
        y = example['input'][None].to(device)
        # Stable per-pair noise seed makes cache resumption exactly reproducible.
        pair_seed = int.from_bytes(hashlib.blake2b(f'{cat}/{name}|{args.seed}'.encode(),
                                                   digest_size=8).digest(), 'little') % (2**63)
        generator = torch.Generator(device=device).manual_seed(pair_seed)
        noise = torch.randn(y.shape, device=device, generator=generator)
        with torch.inference_mode():
            pseudo = diffusion.sample(teacher, y, steps=args.sample_steps, noise=noise)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + '.tmp.png')
        rgb_to_image(pseudo[0]).save(tmp)
        os.replace(tmp, dest)
        done += 1
    print(f'Finished teacher cache: new={done}, reused={skipped}, total={len(train)} in {cache}', flush=True)


def cmd_train_student(args):
    _train_checks(args)
    seed_everything(args.seed)
    device = device_from_args(args.device)
    amp = bool(args.amp and device.type == 'cuda')
    out = Path(args.outdir)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output folder not empty: {out}. Use --resume or new --outdir')
    out.mkdir(parents=True, exist_ok=True)
    ds = get_dataset(args, 'train', crop=args.crop_size, fixed_crop=True, augment=False)
    cache_dir = Path(args.cache_dir)
    metadata = cache_dir / 'metadata.json'
    if not metadata.exists():
        raise FileNotFoundError(f'Missing teacher cache metadata: {metadata}; run cache-teacher')
    cache_cfg = json.loads(metadata.read_text())
    # Exact crop and split alignment are important for paired KD targets.
    if (cache_cfg['crop_size'] != args.crop_size or cache_cfg['seed'] != args.seed or
            cache_cfg['data_root'] != str(Path(args.data_root).expanduser().resolve())):
        raise ValueError('Training data/seed/crop do not match teacher pseudo-target cache')
    tr = DistillationDataset(ds, cache_dir, augment=True)
    val = get_dataset(args, 'val', crop=None)
    tr_dl = loader(tr, args.batch_size, args.workers, True)
    vl_dl = loader(val, 1, args.workers, False)
    student = MobileIEStudent(args.width).to(device)
    config = {'architecture': 'mobileie_deterministic_v1', 'width': args.width,
              'teacher_cache_sha256': cache_cfg['teacher_sha256'],
              'student_noise': False}
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best = _resume_if_any(args.resume, config, student, opt, sch, scaler)
    print(f'STUDENT: params(train)={sum(p.numel() for p in student.parameters()):,}, '
          f'width={args.width}, train={len(tr)}, val={len(val)}, '
          f'GT weight=1, teacher={args.kd_weight}; NO noise input', flush=True)
    for epoch in range(start, args.epochs + 1):
        student.train()
        summ = 0.
        count = 0
        tstart = time.perf_counter()
        bar = tqdm(tr_dl, desc=f'Student epoch {epoch}/{args.epochs}')
        for j, batch in enumerate(bar, 1):
            y = batch['input'].to(device, non_blocking=True)
            gt = batch['target'].to(device, non_blocking=True)
            pseudo = batch['teacher'].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=amp):
                out_img = student(y)  # ONE argument, deterministic. No noise at any step.
                loss = (F.l1_loss(out_img, gt) + args.kd_weight * F.l1_loss(out_img, pseudo)
                        + args.edge_weight * gradient_loss(out_img, gt))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(student.parameters(), 1.)
            scaler.step(opt)
            scaler.update()
            summ += float(loss.detach())
            count += 1
            bar.set_postfix(loss=f'{summ/count:.4f}')
            if args.max_train_batches and j >= args.max_train_batches:
                break
        sch.step()
        measure = (epoch % args.val_every == 0) or (epoch == args.epochs)
        val_psnr = float('nan')
        if measure:
            student.eval()
            rows = evaluate_loader('student', student, None, vl_dl, device,
                                   limit=args.val_limit)
            val_psnr = rows['ALL']['psnr']
            print(f'STUDENT epoch={epoch} val PSNR={val_psnr:.3f} dB '
                  f'({rows["ALL"]["count"]} pairs)', flush=True)
        improved = measure and val_psnr > best
        if improved:
            best = val_psnr
        state = _checkpoint_dict(config, student, opt, sch, scaler, epoch, best)
        safe_save(out / 'last.pt', state)
        if improved:
            safe_save(out / 'best.pt', state)
            write_eval_csv(out / 'best_val_summary.csv', rows)
        history_append(out / 'history.csv', {'epoch': epoch, 'train_loss': summ / max(1, count),
                       'val_psnr': val_psnr, 'best_val_psnr': best,
                       'seconds': round(time.perf_counter() - tstart, 2)})
        print(f'STUDENT {epoch}/{args.epochs} loss={summ/max(1,count):.5f} '
              f'best={best:.3f} dB', flush=True)
    print(f'Student checkpoint: {out / "best.pt"}', flush=True)


def cmd_eval(args):
    device = device_from_args(args.device)
    data = get_dataset(args, args.split, crop=None)
    dl = loader(data, 1, args.workers, False)
    if args.model == 'teacher':
        model, schedule, _ = load_teacher(args.checkpoint, device)
    else:
        model, _ = load_student(args.checkpoint, device)
        schedule = None
    rows = evaluate_loader(args.model, model, schedule, dl, device, teacher_steps=args.sample_steps,
                           seed=args.seed, ssim=True, limit=args.limit)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_eval_csv(out / 'summary.csv', rows)
    for cat, row in rows.items():
        print(f'{cat:16s} N={row["count"]:4d} input={row["input_psnr"]:.3f} '
              f'PSNR={row["psnr"]:.3f} SSIM={row["ssim"]:.4f}', flush=True)
    print(f'Wrote {out / "summary.csv"}', flush=True)


def cmd_infer(args):
    from PIL import Image
    device = device_from_args(args.device)
    if args.model == 'teacher':
        model, schedule, _ = load_teacher(args.checkpoint, device)
    else:
        model, _ = load_student(args.checkpoint, device)
        schedule = None
    inp = Path(args.input)
    files = sorted([p for p in inp.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]) if inp.is_dir() else [inp]
    if not files:
        raise RuntimeError(f'No images found in {inp}')
    out_path = Path(args.output)
    for i, path in enumerate(tqdm(files, desc='Image inference')):
        with Image.open(path) as img:
            y = image_to_tensor(img)[None].to(device)
        with torch.inference_mode():
            if args.model == 'teacher':
                g = torch.Generator(device=device).manual_seed(args.seed + i)
                noise = torch.randn(y.shape, device=device, generator=g)
                output = schedule.sample(model, y, steps=args.sample_steps, noise=noise)
            else:
                output = model(y).clamp(0, 1)   # NO noise or timesteps
        dest = (out_path / path.with_suffix('.png').name) if inp.is_dir() else out_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        rgb_to_image(output[0]).save(dest)
    print(f'Wrote {len(files)} image(s) to {out_path}', flush=True)


def cmd_export(args):
    device = torch.device('cpu')
    model, cfg = load_student(args.checkpoint, device, fuse=True)
    model.eval()

    class ClampedStudent(nn.Module):
        def __init__(self, restorer):
            super().__init__()
            self.restorer = restorer

        def forward(self, degraded):
            return self.restorer(degraded).clamp(0, 1)

    wrapper = ClampedStudent(model)
    x = torch.rand(1, 3, args.size, args.size)
    with torch.inference_mode():
        expected = wrapper(x)
    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('Install export extras: uv sync --extra export') from exc
    torch.onnx.export(wrapper, (x,), str(outfile), input_names=['degraded'],
                      output_names=['restored'], opset_version=17,
                      dynamo=False, export_params=True, do_constant_folding=True)
    graph = onnx.load(str(outfile))
    onnx.checker.check_model(graph)
    inputs = [i.name for i in graph.graph.input]
    if inputs != ['degraded']:
        raise RuntimeError(f'Expected ONE input (degraded), got {inputs}')
    print(f'ONNX checker passed, model inputs={inputs}, output=restored: {outfile}', flush=True)
    maxerr = None
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(outfile), providers=['CPUExecutionProvider'])
        got = session.run(None, {'degraded': x.numpy()})[0]
        maxerr = float((torch.from_numpy(got) - expected).abs().max())
        print(f'ONNX Runtime vs PyTorch max abs error = {maxerr:.6e}', flush=True)
        if maxerr > 1e-4:
            raise RuntimeError('ONNX/PyTorch validation error > 1e-4')
    except ImportError:
        print('onnxruntime not installed; skipped numerical check', flush=True)
    metadata = {'architecture': cfg['architecture'], 'student_noise': False,
                'input': {'name': 'degraded', 'shape': [1, 3, args.size, args.size],
                          'order': 'RGB', 'range': [0, 1], 'dtype': 'float32'},
                'output': {'name': 'restored', 'shape': [1, 3, args.size, args.size],
                           'range': [0, 1]}, 'onnx_vs_pytorch_max_abs_error': maxerr}
    outfile.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f'Wrote metadata: {outfile.with_suffix(".json")}', flush=True)


def cmd_profile(args):
    model = MobileIEStudent(args.width).eval()
    x = torch.rand(1, 3, args.size, args.size)
    with torch.inference_mode():
        raw = model(x)
        fused = model.fused_copy()
        optimized = fused(x)
    conv_macs = 0
    def hook(module, inp, out):
        nonlocal conv_macs
        kernel_h, kernel_w = module.kernel_size
        batch, channels, h, w = out.shape
        conv_macs += (batch * channels * h * w *
                      (module.in_channels // module.groups) * kernel_h * kernel_w)
    handles = [m.register_forward_hook(hook) for m in fused.modules() if isinstance(m, nn.Conv2d)]
    with torch.inference_mode():
        fused(x)
    for h in handles:
        h.remove()
    ntrain = sum(p.numel() for p in model.parameters())
    nfused = sum(p.numel() for p in fused.parameters())
    print(f'width={args.width}, input={args.size}x{args.size}, training params={ntrain:,}, '
          f'fused params={nfused:,}, Conv2d MACs={conv_macs/1e9:.4f} G, '
          f'max fusion difference={(raw-optimized).abs().max().item():.3e}', flush=True)
    print('MACs exclude elementwise ops, resizing, memory transfers and model I/O.', flush=True)


def make_parser():
    p = argparse.ArgumentParser(prog='weatherdiff', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train-teacher', help='Train real multi-step conditional diffusion teacher')
    add_dataset_args(t)
    add_train_args(t)
    t.set_defaults(epochs=80, batch_size=4, lr=2e-4)
    t.add_argument('--base', type=int, default=48)
    t.add_argument('--timesteps', type=int, default=1000)
    t.add_argument('--sample-steps', type=int, default=20, help='DDIM teacher steps during validation')
    t.add_argument('--outdir', default='outputs/teacher')
    t.set_defaults(fn=cmd_train_teacher)

    c = sub.add_parser('cache-teacher', help='Generate deterministic teacher pseudo-labels once')
    add_dataset_args(c)
    c.add_argument('--teacher', default='outputs/teacher/best.pt')
    c.add_argument('--sample-steps', type=int, default=20)
    c.add_argument('--crop-size', type=int, default=256)
    c.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    c.add_argument('--cache-dir', default='outputs/teacher_cache')
    c.set_defaults(fn=cmd_cache_teacher)

    s = sub.add_parser('train-student', help='GT + diffusion-teacher KD, student no noise')
    add_dataset_args(s)
    add_train_args(s)
    s.set_defaults(epochs=40, batch_size=8, lr=2e-4)
    s.add_argument('--width', type=int, default=32)
    s.add_argument('--cache-dir', default='outputs/teacher_cache')
    s.add_argument('--kd-weight', type=float, default=.25)
    s.add_argument('--edge-weight', type=float, default=.1)
    s.add_argument('--outdir', default='outputs/student')
    s.set_defaults(fn=cmd_train_student)

    e = sub.add_parser('evaluate', help='CDD11 PSNR and SSIM')
    add_dataset_args(e)
    e.add_argument('--model', required=True, choices=('teacher', 'student'))
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--split', choices=('val', 'test'), default='test')
    e.add_argument('--sample-steps', type=int, default=20)
    e.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    e.add_argument('--outdir', default='outputs/evaluation')
    e.add_argument('--limit', type=int, default=0, help='0 full set; >0 quick sample')
    e.set_defaults(fn=cmd_eval)

    i = sub.add_parser('infer', help='Image or folder inference (degraded only for student)')
    i.add_argument('--model', required=True, choices=('teacher', 'student'))
    i.add_argument('--checkpoint', required=True)
    i.add_argument('--input', required=True)
    i.add_argument('--output', required=True)
    i.add_argument('--sample-steps', type=int, default=20)
    i.add_argument('--seed', type=int, default=42)
    i.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    i.set_defaults(fn=cmd_infer)

    x = sub.add_parser('export-onnx', help='Export fused deterministic 1-input student')
    x.add_argument('--checkpoint', default='outputs/student/best.pt')
    x.add_argument('--size', type=int, default=256)
    x.add_argument('--output', default='outputs/mobile/student_256.onnx')
    x.set_defaults(fn=cmd_export)

    profile = sub.add_parser('profile', help='Fused params/MACs and fusion correctness')
    profile.add_argument('--width', type=int, default=32)
    profile.add_argument('--size', type=int, default=256)
    profile.set_defaults(fn=cmd_profile)
    return p


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
