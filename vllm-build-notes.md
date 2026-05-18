# vllm Build Notes (Titan V / sm_70)

**Environment:** 4× NVIDIA Titan V (sm_70), CUDA 12.6, Python 3.11, Ubuntu 22.04

---

## Step 1 — Create venv and install torch

```bash
python3.11 -m venv /home/ice/llm/vllm-build-env
source /home/ice/llm/vllm-build-env/bin/activate
pip install --upgrade pip uv

# torch must match requirements/cuda.txt (currently 2.11.0)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

## Step 2 — Install runtime requirements

The `[cu13]` extra in `cuda.txt` is only valid for CUDA ≥ 12.9 — patch it out first:

```bash
cd /home/ice/llm/vllm
sed 's/nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' requirements/cuda.txt | \
  sed 's|-r common.txt|-r /home/ice/llm/vllm/requirements/common.txt|' \
  > /tmp/cuda_req_patched.txt

pip install -r /tmp/cuda_req_patched.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126

# Also install build-time tools
pip install setuptools-scm cmake jinja2
```

## Step 3 — cmake configure (offline, using cached .deps sources)

After the first build, external deps are cached in `/home/ice/llm/vllm/.deps/`.
Point cmake at them via env vars to avoid re-downloading from GitHub.

```bash
mkdir -p /home/ice/llm/vllm/build
rm -rf /home/ice/llm/vllm/build/CMakeCache.txt /home/ice/llm/vllm/build/CMakeFiles

DEPS=/home/ice/llm/vllm/.deps
cd /home/ice/llm/vllm/build && nohup bash -c '
  TORCH_CUDA_ARCH_LIST="7.0 7.0+PTX" \
  PATH="/usr/local/cuda-12.6/bin:$PATH" \
  VLLM_CUTLASS_SRC_DIR=/home/ice/llm/vllm/.deps/cutlass-src \
  TRITON_KERNELS_SRC_DIR=/home/ice/llm/vllm/.deps/triton_kernels-src/python/triton_kernels/triton_kernels \
  FLASH_MLA_SRC_DIR=/home/ice/llm/vllm/.deps/flashmla-src \
  DEEPGEMM_SRC_DIR=/home/ice/llm/vllm/.deps/deepgemm-src \
  QUTLASS_SRC_DIR=/home/ice/llm/vllm/.deps/qutlass-src \
  VLLM_FLASH_ATTN_SRC_DIR=/home/ice/llm/vllm/.deps/vllm-flash-attn-src \
  /home/ice/llm/vllm-build-env/bin/cmake -G Ninja \
    -DVLLM_PYTHON_EXECUTABLE=/home/ice/llm/vllm-build-env/bin/python3 \
    -DCMAKE_INSTALL_PREFIX=/home/ice/llm/vllm \
    -DCMAKE_BUILD_TYPE=Release \
    .. && echo "CMAKE_CONFIGURE_SUCCESS"
' > /home/ice/llm/tmp/vllm_cmake_config.log 2>&1 &
echo "cmake PID: $!"

# Wait for completion (~15s with local deps):
tail -f /home/ice/llm/tmp/vllm_cmake_config.log
# Should end with: "-- Configuring done" and "CMAKE_CONFIGURE_SUCCESS"
```

> **First time (no .deps cache):** cmake will download deps from GitHub.
> Remove the env vars starting with `VLLM_*`/`TRITON_*`/`FLASH_*`/`DEEPGEMM_*`/`QUTLASS_*`
> and cmake will fetch them automatically into `.deps/`. This takes several minutes.

## Step 4 — compile with ninja (resumeable)

```bash
cd /home/ice/llm/vllm/build && nohup bash -c '
  PATH="/usr/local/cuda-12.6/bin:$PATH" \
  NVCC_THREADS=4 \
  ninja -j 10 install
  echo "NINJA_EXIT_CODE=$?"
' > /home/ice/llm/tmp/vllm_build.log 2>&1 &
echo "ninja PID: $!"
```

Monitor progress:
```bash
grep -E "^\[" /home/ice/llm/tmp/vllm_build.log | tail -3   # step counter
tail -3 /home/ice/llm/tmp/vllm_build.log                    # last lines
```

**If interrupted**, just re-run the same `ninja -j 10 install` command — ninja skips
already-compiled `.o` files and resumes from where it left off.

**Total steps:** ~305. Wall time on Titan V with `-j 10`: ~45-90 minutes.

Success indicator: `NINJA_EXIT_CODE=0` at end of log.

> **Note:** Do NOT use `pip install -e .` — it builds in a `/tmp/tmpXXXX` dir
> (not resumeable) and the verbose mode floods stdout causing 100% RAM and crash.

## Step 5 — register the package (no compilation, zero RAM)

After ninja completes, the `.so` files are installed into `/home/ice/llm/vllm/vllm/`.
Register them with the venv using a `.pth` file instead of `pip install -e`:

```bash
SITE=/home/ice/llm/vllm-build-env/lib/python3.11/site-packages

# Add vllm source to Python path
echo "/home/ice/llm/vllm" > $SITE/vllm.pth

# Create minimal dist-info so vllm's internal version detection works
VERSION=$(cd /home/ice/llm/vllm && /home/ice/llm/vllm-build-env/bin/python -c \
  "import setuptools_scm; print(setuptools_scm.get_version())" 2>/dev/null || echo "0.0.1+local")
mkdir -p $SITE/vllm-${VERSION}.dist-info
cat > $SITE/vllm-${VERSION}.dist-info/METADATA << EOF
Metadata-Version: 2.1
Name: vllm
Version: ${VERSION}
EOF
echo "pip" > $SITE/vllm-${VERSION}.dist-info/INSTALLER
echo '{"url": "file:///home/ice/llm/vllm", "dir_info": {"editable": true}}' \
  > $SITE/vllm-${VERSION}.dist-info/direct_url.json
```

## Step 6 — verify

```bash
/home/ice/llm/vllm-build-env/bin/python -c "
import vllm; print('vllm:', vllm.__version__)
import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0), '| CC:', torch.cuda.get_device_capability(0))
from vllm.platforms import current_platform; print('platform:', current_platform)
import importlib.metadata; print('pip metadata:', importlib.metadata.version('vllm'))
"
# Expected output:
# vllm: 0.0.1+local
# torch: 2.11.0+cu126 | CUDA: True
# GPU: NVIDIA TITAN V | CC: (7, 0)
# platform: <vllm.platforms.cuda.NvmlCudaPlatform ...>
# pip metadata: 0.0.1+local
```

---

## Runtime notes for Titan V (sm_70)

| Feature | Status |
|---------|--------|
| FlashAttention | ❌ Requires sm ≥ 8.0 |
| FlashInfer | ❌ Requires sm ≥ 7.5 |
| **Triton attention** | ✅ Works (auto-selected) |
| bfloat16 | ❌ Use `--dtype half` |
| FP8 quantization | ❌ Requires sm ≥ 8.9 |
| TurboQuant KV cache | ✅ Works |
| GGUF models | ✅ Works |

Serving example:
```bash
source /home/ice/llm/vllm-build-env/bin/activate
vllm serve <model_path> --dtype half --enforce-eager
```

## .so files installed by ninja

```
vllm/_C.abi3.so
vllm/_C_stable_libtorch.abi3.so
vllm/_moe_C.abi3.so
vllm/spinloop.abi3.so
vllm/cumem_allocator.abi3.so
vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so
```
