#!/usr/bin/env bash
set -euo pipefail

# ContextFlow LMCache CacheBlend environment setup.
# Verified target:
# - GPU: NVIDIA A100-SXM4-40GB
# - vLLM: 0.18.0+cu130
# - LMCache: 0.4.6.dev6
# - Torch: 2.10.0+cu130
#
# Usage:
#   bash examples/contextflow/setup_cacheblend_env.sh
#   RUN_SMOKE=1 bash examples/contextflow/setup_cacheblend_env.sh
#   RESET_ENV=1 bash examples/contextflow/setup_cacheblend_env.sh

REPO_ROOT="$(git rev-parse --show-toplevel)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
VENV_DIR="$REPO_ROOT/.venv-lmcache-nightly"

cd "$REPO_ROOT"

echo "[1/8] Repo root: $REPO_ROOT"

if [[ "${RESET_ENV:-0}" == "1" ]]; then
  echo "[2/8] RESET_ENV=1, removing old venv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[2/8] Creating venv..."
  uv venv "$VENV_DIR" --python 3.12 --seed --managed-python
else
  echo "[2/8] Reusing existing venv: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/8] Setting runtime environment..."
mkdir -p "$HOME/hf-cache" /tmp/cf

export HF_HOME="$HOME/hf-cache"
export HF_HUB_CACHE="$HOME/hf-cache/hub"
export TOKENIZERS_PARALLELISM=false

export TMPDIR=/tmp/cf
export TMP=/tmp/cf
export TEMP=/tmp/cf

export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_ATTENTION_BACKEND || true

echo "[4/8] Installing vLLM dependencies and CUDA 13.0 vLLM wheel..."
UV_TORCH_BACKEND=cu130 uv pip install "vllm==0.18.0"

uv pip install --force-reinstall --no-deps \
  https://github.com/vllm-project/vllm/releases/download/v0.18.0/vllm-0.18.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl

echo "[5/8] Installing pinned LMCache nightly..."
uv pip install "lmcache==0.4.6.dev6" --pre \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly \
  --index-strategy unsafe-best-match

uv pip install "flashinfer-python==0.6.6" "psutil"

echo "[6/8] Patching vLLM gpu_worker.py..."
cd "$PARENT_DIR"

GPU_WORKER="$(
python - <<'PY' 2>/tmp/vllm_path_log.txt | tail -n 1
import vllm.v1.worker.gpu_worker as gw
print(gw.__file__)
PY
)"

echo "GPU_WORKER=$GPU_WORKER"

python - <<'PY'
from pathlib import Path
import vllm.v1.worker.gpu_worker as gw

p = Path(gw.__file__)
text = p.read_text()

marker = "LMCache CacheBlend model registration patch"

if marker in text:
    print("gpu_worker.py already patched:", p)
else:
    target = "            self.model_runner.load_model(load_dummy_weights=dummy_weights)\n"

    insert = """            self.model_runner.load_model(load_dummy_weights=dummy_weights)

        # LMCache CacheBlend model registration patch.
        # vLLM 0.18 already initializes KV transfer later in initialize_from_config(),
        # so we only register the loaded model here.
        from lmcache.v1.compute.models.utils import VLLMModelTracker
        from lmcache.integration.vllm.utils import ENGINE_NAME

        VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
        logger.info("LMCache CacheBlend registered vLLM model: %s", ENGINE_NAME)
"""

    if target not in text:
        raise RuntimeError("Could not find model_runner.load_model anchor in gpu_worker.py.")

    backup = p.with_suffix(p.suffix + ".bak.contextflow_cacheblend")
    backup.write_text(text)

    text = text.replace(target, insert, 1)
    p.write_text(text)

    print("Patched gpu_worker.py:", p)
    print("Backup:", backup)
PY

python -m py_compile "$GPU_WORKER"

echo "[7/8] Patching LMCache flash_attn.py vLLM Attention import path..."

LMCACHE_FLASH_ATTN="$(
python - <<'PY' 2>/tmp/lmcache_path_log.txt | tail -n 1
import lmcache
from pathlib import Path
print(Path(lmcache.__file__).parent / "v1/compute/attention/flash_attn.py")
PY
)"

echo "LMCACHE_FLASH_ATTN=$LMCACHE_FLASH_ATTN"

python - <<'PY'
from pathlib import Path
import lmcache

p = Path(lmcache.__file__).parent / "v1/compute/attention/flash_attn.py"
text = p.read_text()

old = "from vllm.attention import Attention\n"
new = """try:
    from vllm.attention import Attention
except ModuleNotFoundError:
    from vllm.model_executor.layers.attention import Attention
"""

if "vllm.model_executor.layers.attention import Attention" in text:
    print("flash_attn.py already patched:", p)
else:
    if old not in text:
        raise RuntimeError("Could not find old vllm.attention import in flash_attn.py.")

    backup = p.with_suffix(p.suffix + ".bak.contextflow_attention_path")
    backup.write_text(text)

    text = text.replace(old, new, 1)
    p.write_text(text)

    print("Patched flash_attn.py:", p)
    print("Backup:", backup)
PY

python -m py_compile "$LMCACHE_FLASH_ATTN"

echo "[8/8] Verifying environment..."
cd "$PARENT_DIR"

python - <<'PY'
import inspect
import importlib.metadata as md
import torch, lmcache, vllm

print("=== Versions ===")
for p in ["lmcache", "vllm", "torch", "transformers", "numpy", "flashinfer-python"]:
    try:
        print(p, md.version(p))
    except Exception:
        print(p, "NOT FOUND")

print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("lmcache file:", lmcache.__file__)
print("vllm file:", vllm.__file__)

from vllm.model_executor.layers.rotary_embedding import get_rope
sig = inspect.signature(get_rope)
print("get_rope:", sig)
assert "rope_parameters" in sig.parameters, "vLLM get_rope must support rope_parameters"

from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from vllm.model_executor.layers.attention import Attention

print("LMCBlenderBuilder OK:", LMCBlenderBuilder)
print("VLLMModelTracker OK:", VLLMModelTracker)
print("ENGINE_NAME:", ENGINE_NAME)
print("attention utils OK:", infer_attn_backend_from_vllm)
print("vLLM Attention path OK:", Attention)
PY

echo
echo "Setup complete."
echo "Activate later with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Run official LMCache CacheBlend example:"
echo "  cd $PARENT_DIR"
echo "  python LMCache/examples/blend_kv_v1/blend.py --model Qwen/Qwen2.5-0.5B-Instruct"
echo "  python LMCache/examples/blend_kv_v1/blend.py --model Qwen/Qwen2.5-7B-Instruct"

if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
  echo
  echo "RUN_SMOKE=1, running Qwen2.5-0.5B official blend smoke..."
  rm -rf /tmp/cf/*
  mkdir -p /tmp/cf
  cd "$PARENT_DIR"
  python "$REPO_ROOT/examples/blend_kv_v1/blend.py" \
    --model Qwen/Qwen2.5-0.5B-Instruct
fi
