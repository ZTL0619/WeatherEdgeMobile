# MobileIE-OneStep Android CPU 测速补丁

## 目的

将 WeatherEdgeMobile 里现有的 UDBM-S 单图 Benchmark 入口**替换**为 MobileIE-OneStep。
**不更改**原有 YOLO26 相机、单图、视频检测和 ONNX/NCNN 的逻辑。

这次测的是真正从 CDD11 训练完成的 `results/student_mobileie/distill/best.pt` 导出的**单次去噪**模型，**不是**训练阶段 235795 参数的多分支模型：导出时自动融合为 121107 参数的部署模型。

## 第一步：Ubuntu 导出 (从项目根目录)

```bash
cd ~/Uncertainty-dl-test
uv pip install onnx onnxscript
uv run python export_mobileie_student_onnx.py \
    --checkpoint results/student_mobileie/distill/best.pt \
    --output results/student_mobileie/mobileie_onestep_256.onnx \
    --size 256
```

导出成功打印 `ONNX checker passed`。如有报错，不要继续 Android 步骤。

## 第二步：拷贝模型到 Windows

用 WinSCP 将 Ubuntu 下的 `results/student_mobileie/mobileie_onestep_256.onnx` 放到 Windows 的：

`C:\Users\admin\Desktop\projects\WeatherEdgeMobile\app\src\main\assets\mobileie_onestep_256.onnx`

注意：**文件名必须完全一致**，Android 项目 `assets/` 目录，不是 Python 项目目录。

## 第三步：安装 Kotlin 补丁 (Windows PowerShell)

把本 ZIP **解压到** `C:\Users\admin\Desktop\projects\WeatherEdgeMobile\`，保留 `app/src/main/java/.../MobileIeBenchmark.kt` 子目录。

在 PowerShell 执行：

```powershell
cd C:\Users\admin\Desktop\projects\WeatherEdgeMobile
py install_mobileie_android.py
```

这条安装命令会自动备份本机的 MainActivity，随后只替换已有 UDBM Benchmark 的文字、类名和入口变量，复制独立的 `MobileIeBenchmark.kt`。不修改 YOLO 源码。若检测不到预期旧入口，它会停止而不是覆盖不匹配的 MainActivity。

`app/build.gradle.kts` 之前已添加 `implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")`，保持不变。

## 第四步：编译运行

Android Studio -> `Build -> Assemble Project` -> `Run app` -> 点击 `MobileIE-1Step | Single-image benchmark` -> 选本地一张图片。

界面显示左右恢复效果 + Mean/P50/P90/FPS。

## 测速口径

- 图像和随机噪声：两个 `float32 NCHW [1,3,256,256]` 输入，名称分别是 `degraded` 和 `noise`。
- 每张图只加一次噪声，运行一个 MobileIE 去噪器。ONNX 图包含 `x1=degraded+kappa*noise`。
- ONNX Runtime `CPU Execution Provider`，CPU 4 线程。
- 预热 1 次 + 计时 5 次，与之前 UDBM-S 196.4ms 测试协议相同；可以进行初步对比（设备状态可能不同）。
- 耗时仅统计 `OrtSession.run`，不包含模型加载、读图、缩放、张量准备、显示，模型加载另列。
- 使用固定随机数作为输入以控制测试变量；不重复生成噪声。
- **Android 模拟器结果不等于真机延迟**。后续手机实测需使用相同模型和测量协议。
- 模型按 256x256 固定空间尺寸导出。此阶段仅测速+预览，不代表原始大分辨率视频性能。

## 故障排除

1. 报 `Model asset missing`：检查 `app/src/main/assets/mobileie_onestep_256.onnx` 是否存在，重建 APK。
2. 报 `ONNX input mismatch`：确认使用的是 MobileIE 导出的 `mobileie_onestep_256.onnx`，不要混入旧 `udbm_s_256.onnx`。
3. 报 `current MainActivity.kt is not the expected`：停止操作，提供本机 MainActivity 源码以做适配，不覆盖本地修改。
4. Tensor shape 错：确认 `--size 256` 并使用导出得到的 `distill/best.pt`。
