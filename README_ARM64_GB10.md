# 在 ARM64 + NVIDIA GB10 上編譯 SageAttention 2.2.0

本指南記錄了在 **NVIDIA DGX Spark (Grace CPU + GB10 GPU)** 上成功編譯 SageAttention 的完整過程。

## 硬體與軟體環境

| 項目 | 規格 |
|------|------|
| **CPU** | NVIDIA Grace (ARM64/aarch64) |
| **GPU** | NVIDIA GB10 (Blackwell 架構) |
| **Compute Capability** | 12.1 (SM121) |
| **CUDA Toolkit** | 13.0 |
| **Python** | 3.10 |
| **PyTorch** | 2.9.1+cu130 |
| **Triton** | 3.5.1 |
| **作業系統** | Linux (aarch64) |

## 預編譯 Wheel 下載

如果你的環境與上述相同，可以直接下載預編譯的 wheel 檔案：

👉 **[下載 Release](https://github.com/GarfieldHuang/SageAttention/releases/tag/v2.2.0-arm64-gb10)**

```bash
# 下載後安裝
wget https://github.com/GarfieldHuang/SageAttention/releases/download/v2.2.0-arm64-gb10/sageattention-2.2.0-cp310-cp310-linux_aarch64.whl

pip install sageattention-2.2.0-cp310-cp310-linux_aarch64.whl
```

## 從源碼編譯步驟

### 1. 確認環境

```bash
# 確認 CUDA 版本 (需要 >= 12.8)
nvcc --version

# 確認 GPU 資訊
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

預期輸出：
```
name, compute_cap, driver_version
NVIDIA GB10, 12.1, 580.95.05
```

### 2. 建立 Conda 環境

```bash
# 建立 Python 3.10 環境
conda create -n sage_arm64 python=3.10 -y
conda activate sage_arm64
```

### 3. 安裝 PyTorch 和依賴

```bash
# 安裝 PyTorch 2.9.1+cu130 (ARM64 版本)
pip install torch==2.9.1+cu130 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

# 安裝編譯所需依賴
pip install ninja packaging einops
```

### 4. 克隆 SageAttention

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
```

### 5. 設定環境變數並編譯

```bash
# 設定 CUDA 環境
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.1"
export MAX_JOBS=32
export EXT_PARALLEL=4

# 編譯安裝
pip install . --no-build-isolation

# 或者建立 wheel 檔案
pip wheel . --no-build-isolation -w ./dist/
```

### 6. 驗證安裝

```bash
# 切換到其他目錄避免 import 衝突
cd /tmp

python -c "
import torch
from sageattention import sageattn

print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name()}')

# 測試運算
q = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')

output = sageattn(q, k, v, tensor_layout='HND')
print(f'輸出形狀: {output.shape}')
print('✓ SageAttention 運作正常！')
"
```

## 常見問題

### Q1: PyTorch 顯示 "cuda capability 12.1 not supported" 警告

這只是警告，不影響實際運行。PyTorch 2.9.1 發布時 Blackwell 架構尚新，官方還未更新支援清單，但實際上可以正常使用。

### Q2: 為什麼不用 cu128？

雖然 CUDA 向下相容，但 **PyTorch 擴展需要與主 PyTorch 使用相同的 CUDA 版本編譯**。如果你的應用（如 ComfyUI）使用 cu130 版本的 PyTorch，SageAttention 也需要用 cu130 編譯。

### Q3: 編譯時出現 "circular import" 錯誤

這通常是因為在源碼目錄內執行 Python。請切換到其他目錄再測試：

```bash
cd /tmp
python -c "from sageattention import sageattn; print('OK')"
```

### Q4: SageAttention3 (FP4) 支援嗎？

SageAttention3 需要 **Python 3.13+** 和 **PyTorch 2.8.0+**。如果你使用 Python 3.10，只能使用 SageAttention 2.2.0（仍支援 SM121）。

## 環境版本對照表

| SageAttention 版本 | Python | PyTorch | CUDA | 備註 |
|-------------------|--------|---------|------|------|
| 2.2.0 | ≥ 3.9 | ≥ 2.3.0 | ≥ 12.8 | 本指南使用此版本 |
| 3.x (Blackwell FP4) | ≥ 3.13 | ≥ 2.8.0 | ≥ 12.8 | 需要更高版本 Python |

## 在 ComfyUI 中使用

如果你的 ComfyUI 環境與編譯環境使用相同的 PyTorch 版本：

```bash
# 在 ComfyUI 環境中安裝
conda activate comfyui-env
pip install /path/to/sageattention-2.2.0-cp310-cp310-linux_aarch64.whl
```

## 編譯產出

本次編譯產生的 wheel 檔案：
- `sageattention-2.2.0-cp310-cp310-linux_aarch64.whl` (約 14MB)

## 參考連結

- [SageAttention 官方倉庫](https://github.com/thu-ml/SageAttention)
- [PyTorch ARM64 Wheel](https://download.pytorch.org/whl/cu130/)
- [NVIDIA Compute Capability](https://developer.nvidia.com/cuda-gpus)

## 貢獻者

- 編譯測試：[@GarfieldHuang](https://github.com/GarfieldHuang)
- 日期：2025年11月30日
- 硬體：NVIDIA DGX Spark (GB10)

---

**License**: MIT (遵循原 SageAttention 授權)
