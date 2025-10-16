# TRELLIS on WSL2 — 故障手记（10.15-10.16踩坑全记录）

> 目标：在 **Windows 11 + WSL2 (Ubuntu 22.04)** 下，**离线/半离线**运行 `microsoft/TRELLIS` 的 Gradio Web Demo，并尽量与官方环境一致。本文记录这两天遇到的所有问题、原因分析与可复现的处理手段，便于后来者复用。

---

## 环境与前置

* **宿主**：Windows 11 + WSL2（Ubuntu 22.04）
* **CUDA**：WSL2 下 CUDA 11.8（`nvcc -V` 显示 `release 11.8`）
* **Conda**：Miniforge / Mambaforge（统一使用 conda 环境）
* **PyTorch**：`torch==2.4.0+cu118`
* **仓库路径**：`/mnt/d/dev/TRELLIS`
* **模型（离线）**：`/mnt/d/models/TRELLIS-image-large/{pipeline.json, ckpts/*.safetensors}`
* **缓存**：`/mnt/d/hf_cache`（含 `facebookresearch/dinov2` 源码）
* **网络**：系统有代理；WSL 提示 “NAT 模式下不支持 localhost 代理”

---

## 故障清单 & 解决方案


### 1) venv / conda 混用 → 包找不到

**现象**

```
ModuleNotFoundError: No module named 'gradio'
```

**原因**
起初用 venv，后面切 conda，PATH 混乱，`python` 指向不同解释器。

**解决（统一用 conda）**

```bash
conda create -n trellis python=3.10 -y
conda activate trellis
```

---

### 2) CUDA 扩展三件套：spconv / kaolin / nvdiffrast

#### 2.1 spconv

**报错**

```
ModuleNotFoundError: No module named 'spconv'
```

**解决**（与 torch2.4/cu118 实测可用）

```bash
pip install "spconv-cu120==2.3.6"
```

#### 2.2 kaolin

**报错 1（占位轮子）**

```
ImportError: This is the kaolin placeholder wheel...
```

**报错 2（缺 Cython）**

```
Kaolin requires Cython >= 0.29.37
```

**报错 3（编译 exit status 137）**
内存/Swap 不足，Ninja 构建中止。

**解决**

```bash
pip uninstall -y kaolin
pip install -U pip setuptools wheel ninja cython
export MAX_JOBS=1                     # 降低并发
# 确保 WSL 分配足够内存/Swap（建议 16~32G Swap）
pip install --no-cache-dir "git+https://github.com/NVIDIAGameWorks/kaolin.git"
```

#### 2.3 nvdiffrast

**报错**

```
ModuleNotFoundError: No module named 'nvdiffrast'
```

**解决**

```bash
pip install nvdiffrast
```

---

### 3) xFormers / flash-attn 下载与兼容

* **xFormers**：直链下载 403；改用 PyPI 对应版本即可。

  ```bash
  pip install "xformers==0.0.27.post2"
  ```
* **flash-attn**：源码编译失败；或本地 `.whl` 路径写成 `D:\...`（WSL 下无效）。

  * 放到 `/mnt/d/.../xxx.whl` 再 `pip install /mnt/d/.../xxx.whl`
  * 或暂不安装，使用 `ATTN_BACKEND=xformers` 即可跑通 TRELLIS。

---

### 4) Hugging Face 404 / 私有权重访问失败

**报错**

```
RepositoryNotFoundError: 404 ... /ckpts/xxx/.json
```

**原因**
repo_id 写错 / 权限不足；或离线模式仍尝试联网。

**解决**

* 确保本地模型目录完整（`pipeline.json` 引用的 `ckpts/*.safetensors` 存在）
* 离线变量：

  ```bash
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TRELLIS_MODEL_ID="/mnt/d/models/TRELLIS-image-large"
  ```
* 让 `from_pretrained` 只走 **本地路径**，不要写成 HF 的 repo_id。

---

### 5) `torch.hub` 离线加载 DINOv2 失败

**报错**

```
RuntimeError: ... no internet connection and the repo could not be found in the cache
```

**原因**
`torch.hub.load('facebookresearch/dinov2', ...)` 默认联网拉 GitHub。

**解决（至少一条）**

1. **预热本地缓存**：把 `facebookresearch/dinov2` 源码放到：

   ```
   /mnt/d/hf_cache/hub/facebookresearch_dinov2_main
   ```

   并设：

   ```bash
   export TORCH_HOME=/mnt/d/hf_cache
   ```
2. **Monkey-patch `torch.hub.load`**：当入参是 `'facebookresearch/dinov2'` 时，强制改为本地目录 `DINOV2_DIR`。
3. **改 TRELLIS 源码**：将 `torch.hub.load('facebookresearch/dinov2', ...)` 改为 `torch.hub.load(<本地 hubconf.py 目录>, ...)`。

---

### 6) Gradio：localhost 不可达 / 端口占用 / 代理干扰

**报错 A**

```
ValueError: When localhost is not accessible, a shareable link must be created.
```

**原因**
系统代理拦截 `localhost`；或 `/etc/hosts` 缺 `127.0.0.1 localhost`。

**解决**

```bash
# /etc/hosts 补齐
grep -E '127\.0\.0\.1\s+localhost|::1\s+localhost' /etc/hosts || \
  sudo bash -c 'echo -e "127.0.0.1\tlocalhost\n::1\tlocalhost" >> /etc/hosts'

# 运行时彻底禁用代理
NO_PROXY=127.0.0.1,localhost,::1 \
no_proxy=127.0.0.1,localhost,::1 \
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= all_proxy= \
python -u run_web.py
```

**报错 B**

```
OSError: Cannot find empty port in range: 7860-7860
```

**解决**

```bash
ss -ltnp | grep :7860
sudo fuser -k 7860/tcp
export GRADIO_SERVER_PORT=7861
```

**本机回环自测**

```bash
python -m http.server 7861 &
curl -I http://127.0.0.1:7861
kill %1
```

---

### 7) Gradio JSON-Schema 解析 Bug

**报错**

```
TypeError: argument of type 'bool' is not iterable
APIInfoParseError: Cannot parse schema True
```

**原因**
某些 Gradio 版本处理布尔型 `additionalProperties: true/false` 有 Bug。

**解决**

* 给 `gradio_client.utils` 做轻量 **monkey-patch**：遇到布尔 schema 时视作 `typing.Any`。
* 或保持 `gradio` 与 `gradio_client` 同一稳定版本（如 4.25.x/4.44.x 组合），必要时仍用补丁兜底。

---

### 8) “NAT 模式不支持 localhost 代理” 警告

**提示**

```
wsl: 检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理。
```

**影响**
`localhost` 被代理影响，Gradio 要求 `share=True`。

**解决**
运行时 **清空代理**（见第 7 节）；或在 Windows 端给 `localhost/127.0.0.1` 加白名单/临时关闭代理。

---

### 9) 构建/运行卡住（资源不足）

**现象**
编译长时间无响应或 `ninja ... exit status 137`。

**原因**
WSL2 内存/Swap 不足；首次加载权重/渲染较慢。

**解决**

* 提高 WSL 内存与 Swap（`.wslconfig` 配置后 `wsl --shutdown` 重启）
* 编译降并发：`export MAX_JOBS=1`
* 关闭无关进程；首次加载耐心等待

---

## 关键检查点（已过）

* CUDA 就绪：

  ```bash
  python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
  ```
* 扩展可导：

  ```python
  import spconv, nvdiffrast.torch as dr, kaolin, xformers
  ```
* 本地权重存在且可读（示例）：

  ```python
  from safetensors.torch import load_file
  t = load_file("/mnt/d/models/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors")
  print(len(t))
  ```
* DINOv2 源码在本地：`/mnt/d/hf_cache/hub/facebookresearch_dinov2_main`

---

## 一键离线运行（当前使用）

> 假设已经在 `run_web.py` 注入了：
>
> * `torch.hub.load` 的离线代理逻辑（指向 `DINOV2_DIR`）
> * Gradio JSON-Schema 的布尔兜底补丁
> * `demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_SERVER_PORT","7861")))`

```bash
# 释放端口
sudo fuser -k 7861/tcp 2>/dev/null || true

# 离线 + 本地
export TRELLIS_MODEL_ID="/mnt/d/models/TRELLIS-image-large"
export DINOV2_DIR="/mnt/d/hf_cache/hub/facebookresearch_dinov2_main"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=false
export TORCH_HOME="/mnt/d/hf_cache"
export GRADIO_SERVER_PORT=7861

# 禁用代理，确保 localhost 可达
NO_PROXY=127.0.0.1,localhost,::1 \
no_proxy=127.0.0.1,localhost,::1 \
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= all_proxy= \
python -u run_web.py
```

出现：

```
[BOOT] Pipeline loaded in XXs
Running on local URL:  http://127.0.0.1:7861
```

浏览器打开 `http://127.0.0.1:7861`。

---

## 常用自检脚本

```bash
python - <<'PY'
def check(tag, fn):
    try:
        fn(); print(tag, "OK")
    except Exception as e:
        print(tag, "ERR ->", e)

check("torch/cuda", lambda: ( __import__("torch"), print("cuda:", __import__("torch").cuda.is_available()) ))
check("spconv",    lambda: __import__("spconv"))
check("nvdiffrast",lambda: __import__("nvdiffrast.torch", fromlist=["dr"]))
check("kaolin",    lambda: __import__("kaolin"))
check("xformers",  lambda: ( __import__("xformers"), __import__("xformers.ops") ))
PY
```

---

## TODO（后续优化）

* 将 `trellis/pipelines/trellis_image_to_3d.py` 中 DINOv2 的 `torch.hub.load('facebookresearch/dinov2', ...)` **正式改为本地路径**，去掉 monkey-patch。
* 固定一组已验证的 `gradio`/`gradio_client` 版本组合，减少 schema 解析变更带来的不稳定。
* 完全离线镜像第三方依赖与权重，提供脚本一键下载/校验。
* 为 WSL 配置合理的 `.wslconfig`（内存/Swap/处理器核心数）。

---



