import torch
from weatherdiff.diffusion import GaussianDiffusion, ConditionalDiffusionUNet
from weatherdiff.student import MobileIEStudent
from weatherdiff.metrics import psnr_batch, ssim_batch


def test_student_deterministic_and_one_input():
    torch.manual_seed(1)
    model = MobileIEStudent(width=16).eval()
    y = torch.rand(1, 3, 24, 28)
    with torch.no_grad():
        a, b = model(y), model(y)
    assert torch.equal(a, b)
    assert a.shape == y.shape
    assert list(model.forward.__code__.co_varnames[:2]) == ['self', 'degraded']


def test_student_branch_fusion_nontrivial():
    model = MobileIEStudent(width=16).eval()
    with torch.no_grad():
        model.out.weight.normal_(0, 0.003)
        model.out.bias.fill_(0.01)
        x = torch.randn(1, 3, 32, 32)
        unfused = model(x)
        fused_model = model.fused_copy()
        fused = fused_model(x)
    assert (unfused - fused).abs().max().item() < 1e-5
    assert sum(p.numel() for p in fused_model.parameters()) < sum(p.numel() for p in model.parameters())


def test_diffusion_clean_gt_not_condition():
    d = GaussianDiffusion(timesteps=20)
    x0 = torch.zeros(1, 3, 8, 8)
    y = torch.ones_like(x0)
    t = torch.tensor([4])
    eps = torch.zeros_like(x0)
    xt = d.q_sample(x0, t, eps)
    assert torch.equal(xt, x0)  # Clean GT only, degraded y ignored by q_sample
    assert not torch.equal(xt, y)


def test_ddim_real_multistep_and_seeded():
    d = GaussianDiffusion(timesteps=20)
    model = ConditionalDiffusionUNet(base=16).eval()
    y = torch.rand(1, 3, 16, 16)
    noise = torch.randn_like(y)
    calls = []
    handle = model.register_forward_hook(lambda m, i, o: calls.append(1))
    with torch.no_grad():
        a = d.sample(model, y, steps=3, noise=noise)
        b = d.sample(model, y, steps=3, noise=noise)
    handle.remove()
    assert len(calls) == 6
    assert torch.equal(a, b)
    assert a.shape == y.shape


def test_metrics_identical():
    y = torch.rand(2, 3, 20, 20)
    assert (psnr_batch(y, y) > 100).all()
    assert (ssim_batch(y, y) > .999).all()
