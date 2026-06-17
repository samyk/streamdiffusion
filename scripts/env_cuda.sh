#!/usr/bin/env bash

# Make pip-installed NVIDIA runtime libraries visible to torch/TensorRT.
if [[ -z "${PYTHON:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  # shellcheck source=common.sh
  source "${ROOT}/scripts/common.sh"
fi

_cuda_lib_dirs() {
  "${PYTHON}" - <<'PY'
import importlib.util
import os

names = (
    "nvidia.cudnn",
    "nvidia.cublas",
    "nvidia.cuda_runtime",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.cusolver",
    "nvidia.cusparse",
    "nvidia.nvjitlink",
    "nvidia.nccl",
    "nvidia.nvtx",
    "nvidia.cufile",
    "nvidia.cuda_nvrtc",
    "nvidia.cusparselt",
    "nvidia.nvshmem",
)
paths: list[str] = []
for name in names:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        continue
    base = spec.submodule_search_locations[0]
    lib_dir = os.path.join(base, "lib")
    if os.path.isdir(lib_dir):
        paths.append(lib_dir)
print(":".join(paths))
PY
}

_CUDA_LIB_DIRS="$(_cuda_lib_dirs)"
if [[ -n "${_CUDA_LIB_DIRS}" ]]; then
  export LD_LIBRARY_PATH="${_CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Hugging Face gated models (FLUX.2 Klein 9B, etc.) need a token in non-interactive sessions.
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
  elif [[ -f "${HOME}/.huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.huggingface/token")"
    export HF_TOKEN
  fi
fi
