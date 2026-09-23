#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $(uname -s) != Linux ]]; then
    echo "This setup task is for WSL/Linux only." >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Run 'mise run setup' first." >&2
    exit 1
fi
if [[ ! -x /usr/local/cuda-12.8/bin/nvcc || ! -x /usr/bin/g++-14 ]]; then
    echo "Install CUDA toolkit 12.8 and gcc-14/g++-14 in WSL first." >&2
    exit 1
fi

# Keep the shared lock and macOS process task intact on this machine.
# The pinned, hash-verified RacketVision checkpoints use MMEngine's legacy load.
if [[ ! -e .mise.local.toml ]]; then
    cat > .mise.local.toml <<'TOML'
[env]
UV_NO_SYNC = "1"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
TOML
fi

if uv run --no-sync python - <<'PY'
import numpy
import torch
import torchvision
from mmcv.ops import get_compiling_cuda_version

assert torch.__version__ == "2.7.1+cu128"
assert torchvision.__version__ == "0.22.1+cu128"
assert numpy.__version__ == "1.26.4"
assert get_compiling_cuda_version().startswith("12.8")
PY
then
    echo "WSL CUDA packages are already installed."
    exit 0
fi

uv pip install --python .venv/bin/python \
    'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python --no-deps 'numpy==1.26.4'
uv pip install --python .venv/bin/python wheel ninja psutil

# CUDA 12.8's math declarations conflict with newer WSL glibc headers.
# Patch a disposable header copy for this build; leave the system toolkit alone.
compat_root=$(mktemp -d /tmp/tennis-cuda-12.8.XXXXXX)
trap 'rm -rf "$compat_root"' EXIT
mkdir -p "$compat_root/include"
cp -a /usr/local/cuda-12.8/targets/x86_64-linux/include/. "$compat_root/include/"
ln -s /usr/local/cuda-12.8/bin "$compat_root/bin"
ln -s /usr/local/cuda-12.8/lib64 "$compat_root/lib64"

uv run --no-sync python - "$compat_root/include/crt/math_functions.h" <<'PY'
from pathlib import Path
import re
import sys

header = Path(sys.argv[1])
source = header.read_text()
for name in ("rsqrt", "rsqrtf", "sinpi", "sinpif", "cospi", "cospif"):
    pattern = rf'(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ [^;\n]+\b{name}\([^;\n]+\))\s*;'
    source, count = re.subn(pattern, r'\1 noexcept (true);', source)
    if count != 1:
        raise SystemExit(f"Expected one CUDA declaration for {name}, found {count}")
header.write_text(source)
PY

CUDA_HOME="$compat_root" CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 \
    CUDAHOSTCXX=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=12.0 \
    MMCV_WITH_OPS=1 MAX_JOBS=2 uv pip install \
    --python .venv/bin/python --no-config --no-deps --no-build-isolation \
    --no-binary mmcv --reinstall-package mmcv 'mmcv==2.1.0'

uv run --no-sync python -c 'from mmcv.ops import nms; print("MMCV CUDA extension is ready")'
