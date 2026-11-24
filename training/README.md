# WeatherDiffusionMobile — 条件扩散教师 + 无噪声 MobileIE 学生

**独立新工程**，不依赖旧 ResShift、UDBM-S、`Uncertainty-dl-test` 或旧 Fusion Teacher checkpoint。

## 设计承诺

| 部分 | 输入 | 训练 | 推理 |
|---|---|---|---|
| **教师** | 扩散中间状态 `x_t`、退化图 `y`、时间步 `t` | 条件 DDPM，随机 `t`，**只对 GT `x0` 进行前向加噪**，预测 `x0` | 从高斯 `x_T` 开始，在 `y` 条件下使用 **20 步 DDIM**（可调整） |
| **学生** | **仅退化图 `y`** | GT 的 L1 + 教师恢复图 L1 + 边缘损失 | **一次 CNN 前向、无噪声、无时间步、无扩散采样** |

**原理说明：** 教师具有严格定义的前向扩散和多步反向采样；学生学到扩散教师的恢复结果，但**不是单步扩散采样器**。这正是当前选择的“多步扩散教师 → 直接恢复学生”方案。不要将本系统称作“学生单步扩散模型”。

训练公式（像素归一化到 [-1,1]）：

```
t ~ Uniform({0,...,T-1}), ε ~ N(0,I)
x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * ε
x0_hat = DiffusionTeacher(x_t, y, t)
L_teacher = L1(x0_hat, x0) + 0.1 * MSE(x0_hat, x0)
```

余弦噪声调度 `T=1000`。教师以 `x0` 为参数化预测目标，通过确定性 DDIM（η=0）逐步更新 `x_t`。推理从独立高斯 `x_T` 开始；**退化图像 `y` 从来不被加噪**。

学生：

```
restored = MobileIEStudent(y)   # y shape [B,3,H,W], RGB in [0,1]
L_student = L1(restored, GT) + 0.25 * L1(restored, diffusion_teacher_GT)
            + 0.1 * gradient_L1(restored, GT)
```

MobileIE-inspired 的 MBRConv/FST/HDPA + 三尺度结构，**不是论文原始 MobileIE 的逐层复现**。MBRConv 在部署时融合成单个标准卷积；默认 width=32 的部署参数量约 0.12M。无 Depthwise Conv。GT 优先于教师的蒸馏权重，防止不成熟的扩散教师带偏学生。

## 0. 使用 uv 安装环境（Ubuntu）

```bash
# 如果没有 uv：
curl -LsSf https://astral.sh/uv/install.sh | sh

# 将 WeatherDiffusionMobile.zip 解压，然后：
cd WeatherDiffusionMobile
uv python install 3.11
uv sync --extra export --extra test
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- `uv sync` 会自动创建 `.venv/` 并安装 PyTorch、Pillow、NumPy、tqdm；`--extra export` 安装 ONNX/ONNX Runtime；`--extra test` 安装 pytest。
- **RTX 4090 训练前必须确认** `torch.cuda.is_available()` 输出 `True`。该环境从 PyPI 安装 PyTorch，Linux CUDA 轮子还需要服务器有兼容 NVIDIA 驱动；若输出 `False`，应先排查驱动或使用 PyTorch 官方 CUDA 对应 wheel，不能继续假定 GPU 可用。
- `uv` 会在首次同步时生成 `uv.lock`。没有将锁文件硬编码为某台服务器的 CUDA 版本。

## 1. CDD11 数据组织

```text
uncertainty/
├── splits/
│   ├── train.txt
│   └── val.txt
├── train/
│   ├── clear/      (GT)
│   ├── haze/
│   ├── rain/
│   └── ... 其他退化类别
└── test/
    ├── clear/
    ├── haze/
    └── ...
```

默认可通过环境变量 `CDD11_ROOT` 设置（**不是**到 `train/` 或 `test/` 子目录）。

```bash
export CDD11_ROOT=/path/to/CDD11
```

首次建立 `outputs/index/cdd11_{train,val,test}.json`，完整解码检查图像并跳过损坏的**图像对**，不删除任何原图。损坏 GT 对应的所有退化配对都会被跳过；仅退化图损坏则只跳过相应配对。报告 `outputs/index/broken_*.csv`。下次启动直接读取索引；源数据发生变化时执行 `--recheck-images` 重新扫描。

**不要用测试集选择网络、修改超参数或蒸馏权重；测试集应留作最终报告。**

## 2. 先做功能检查

```bash
uv run --extra test pytest -q
uv run weatherdiff profile --width 32 --size 256
uv run weatherdiff --help
```

## 3. 完整训练：按阶段执行（推荐，便于检查）

### 第一步：训练真正的条件扩散教师

```bash
uv run weatherdiff train-teacher \
  --data-root "$CDD11_ROOT" \
  --base 48 --timesteps 1000 --sample-steps 20 \
  --crop-size 256 --batch-size 4 --epochs 80 --val-every 5 \
  --device cuda:0 --outdir outputs/teacher
```

教师**从头训练**，不能使用旧的 `checkpoints/teacher_fusion_final.pt`，那是非扩散教师，架构不兼容。

- `outputs/teacher/last.pt`：含 optimizer 的断点恢复文件。
- `outputs/teacher/best.pt`：由验证集（默认全量）PSNR 选出的最佳扩散教师。
- `outputs/teacher/history.csv`：各 epoch 损失和验证 PSNR。
- 训练恢复：在同一命令末尾加 `--resume outputs/teacher/last.pt`，不能同时修改教师 base/timesteps 或训练目标。
- **成本提醒：** 此阶段训练可能明显慢于旧的直接恢复教师；首次教师验证为 20 步 × 验证图片数次网络推理，属于预期行为。训练结果要看实际 CDD11 PSNR，不预先保证超过旧教师。

### 第二步：扩散教师生成一次训练集伪标签

```bash
uv run weatherdiff cache-teacher \
  --data-root "$CDD11_ROOT" \
  --teacher outputs/teacher/best.pt --sample-steps 20 \
  --crop-size 256 --seed 42 \
  --cache-dir outputs/teacher_cache --device cuda:0
```

它会把扩散教师针对每个有效训练样本的固定 256×256 裁剪推理结果写成 RGB PNG，目录如 `outputs/teacher_cache/haze/00001.png`。可以中断、重跑；已生成文件会复用，元数据阻止不同教师 checkpoint 混用。**该过程只执行一次，不在每个学生 epoch 里重复运行扩散教师。** 预计占用一定磁盘空间（与训练对数、图像纹理有关）。

### 第三步：训练确定性 MobileIE 学生

```bash
uv run weatherdiff train-student \
  --data-root "$CDD11_ROOT" \
  --cache-dir outputs/teacher_cache \
  --width 32 --batch-size 8 --epochs 40 \
  --kd-weight 0.25 --edge-weight 0.1 \
  --device cuda:0 --outdir outputs/student
```

训练期间只加载 `y、GT、teacher pseudo-target`，**没有 Gaussian noise、`x_t` 或扩散调度**。产生 `outputs/student/{best.pt,last.pt,history.csv}`。

### 第四步：独立 CDD11 测试集评估

```bash
uv run weatherdiff evaluate \
  --data-root "$CDD11_ROOT" --split test \
  --model student --checkpoint outputs/student/best.pt \
  --device cuda:0 --outdir outputs/eval_student
```

输出 `outputs/eval_student/summary.csv`，每类别及整体的 input PSNR / PSNR / SSIM。

教师本身也可以用相同步数评价：

```bash
uv run weatherdiff evaluate \
  --data-root "$CDD11_ROOT" --split test \
  --model teacher --checkpoint outputs/teacher/best.pt \
  --sample-steps 20 --device cuda:0 --outdir outputs/eval_teacher
```

### 第五步：真实图片推理（只有学生才是端侧候选）

```bash
uv run weatherdiff infer --model student \
  --checkpoint outputs/student/best.pt \
  --input /path/to/degraded.png \
  --output outputs/restored.png --device cuda:0
```

`--input` 为目录时 `--output` 也给目录；支持常见 RGB 图片格式。学生输入只有退化图像，输出相同大小。

### 第六步：导出 Android ONNX（一输入、无噪声）

```bash
uv run --extra export weatherdiff export-onnx \
  --checkpoint outputs/student/best.pt \
  --size 256 --output outputs/mobile/student_256.onnx
```

进行 ONNX checker 及可用时 ONNX Runtime 数值对比，得到 `student_256.onnx` 和同名 `.json`。

- 输入：**仅** `degraded`，float32 `[1,3,256,256]`，RGB，范围 `[0,1]`。
- 输出：`restored`，float32 `[1,3,256,256]`，范围 `[0,1]`。
- 没有 `noise` 第二输入、没有随机种子、没有 DDIM sampler。
- Android 项目适配文件见 `android_integration/`。不需要也不应该把扩散教师装到 Android 上。

## 4. 一条命令串行执行（正式训练可能耗时）

```bash
CDD11_ROOT="$CDD11_ROOT" bash scripts/run_pipeline.sh
```

**不要在已有输出目录运行从头开始的命令以覆盖 checkpoint**。如训练中断，使用对应 `--resume`；缓存过程可直接重复；完成的阶段不应重复跑。

## 5. 检查/性能约束

- 可学习参数量和理论 Conv2d MACs：`weatherdiff profile`。
- 真实 Android 延迟：独立测量 ONNX Runtime `session.run()`（与之前一致 256×256，CPU x4），Mean/P50/P90/FPS；还应另测端到端预后处理和真机。
- 教师预期质量、蒸馏提升、学生 PSNR **尚未在你的真实数据上实测**。代码验证不等于训练完成。
- 如果扩散教师比无扩散教师差，蒸馏也可能降低学生精度。保留 `--kd-weight 0` 的监督基线供对照；这不是自动提升指标的保证。
- 学生不对输入额外加噪，与上一个项目的 `degraded + 0.04 ε` 完全不同。

## 参考与归属

- Ho et al., **Denoising Diffusion Probabilistic Models** (NeurIPS 2020)，DDPM 前向扩散。
- Song et al., **Denoising Diffusion Implicit Models** (ICLR 2021)，DDIM 确定性采样。
- Nichol & Dhariwal, **Improved Denoising Diffusion Probabilistic Models** (ICML 2021)，cosine schedule。
- **MobileIE** (ICCV 2025)，仅借鉴多分支重参数化、特征变换与双路径注意力思想；本代码不是官方 MobileIE 实现。
