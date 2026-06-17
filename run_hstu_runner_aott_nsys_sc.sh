#!/usr/bin/env bash
set -euo pipefail
set -x

REPO=/home/scratch.aleliu_sw/generative-recommenders
BUILD_DIR=/home/scratch.aleliu_sw/hstu_runner_build
RUNNER="$BUILD_DIR/hstu_runner"
PROFILE_ROOT=${1:-$REPO/artifacts/nsys_aott}
STAMP=${NSYS_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="$PROFILE_ROOT/$STAMP"
TMP_ROOT="$RUN_DIR/tmp"
STUB_ROOT="$BUILD_DIR/py_stubs"
FBGEMM_ROOT=/usr/local/lib/python3.12/dist-packages/fbgemm_gpu
TORCH_LIB=/usr/local/lib/python3.12/dist-packages/torch/lib

cd "$REPO"
mkdir -p "$BUILD_DIR" "$RUN_DIR" "$TMP_ROOT" "$STUB_ROOT/security/frameworks/python/exec"
for init in \
  "$STUB_ROOT/security/__init__.py" \
  "$STUB_ROOT/security/frameworks/__init__.py" \
  "$STUB_ROOT/security/frameworks/python/__init__.py" \
  "$STUB_ROOT/security/frameworks/python/exec/__init__.py"; do
  : > "$init"
done
cat > "$STUB_ROOT/security/frameworks/python/exec/subprocess.py" <<'PY'
import subprocess


class TrustedSubprocessWithList:
    @staticmethod
    def run(executable, cmd_args, **kwargs):
        return subprocess.run([executable, *cmd_args], **kwargs)
PY

export PYTHONPATH="$STUB_ROOT:$REPO:${PYTHONPATH:-}"
export TMPDIR="$TMP_ROOT"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$RUN_DIR/cache"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export TRITON_AOT_PATH_PREFIX="$RUN_DIR/triton_aot_compile"
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" \
  "$TRITON_AOT_PATH_PREFIX"

command -v nsys
nsys --version | tee "$RUN_DIR/nsys_version.txt"
nvidia-smi --query-gpu=name --format=csv,noheader | tee "$RUN_DIR/gpu.txt"
python -c 'import torch, fbgemm_gpu, torchrec; print("torch", torch.__version__); print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))' | tee "$RUN_DIR/python_env.txt"

python - <<'PY' > "$BUILD_DIR/compile_env"
import os
import sysconfig
import torch

torch_dir = os.path.dirname(torch.__file__)
print(f'TORCH_DIR={torch_dir}')
print(f'PY_INCLUDE={sysconfig.get_path("include")}')
print(f'ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}')
PY

source "$BUILD_DIR/compile_env"

g++ -std=c++17 -O2 \
  -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
  -I"$PY_INCLUDE" \
  -I"$TORCH_DIR/include" \
  -I"$TORCH_DIR/include/torch/csrc/api/include" \
  generative_recommenders/dlrm_v3/inference/cpp/hstu_runner.cpp \
  -L"$TORCH_DIR/lib" \
  -Wl,-rpath,"$TORCH_DIR/lib" \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -ldl \
  -o "$RUNNER"

LIBPYTHON=$(python - <<'PY'
import os
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR")
ldlibrary = sysconfig.get_config_var("LDLIBRARY")
path = os.path.join(libdir, ldlibrary)
print(path if os.path.exists(path) else "")
PY
)

fbgemm_libs=(
  fbgemm_gpu_config.so
  asmjit.so
  fbgemm.so
  fbgemm_gpu_tbe_common.so
  fbgemm_gpu_tbe_utils.so
  fbgemm_gpu_sparse_async_cumsum.so
  fbgemm_gpu_tbe_index_select.so
  fbgemm_gpu_embedding_inplace_ops.so
  fbgemm_gpu_tbe_inference.so
  fbgemm_gpu_tbe_cache.so
  fbgemm_gpu_tbe_training_forward.so
  fbgemm_gpu_tbe_training_backward.so
  fbgemm_gpu_tbe_training_backward_dense.so
  fbgemm_gpu_tbe_training_backward_split_host.so
  fbgemm_gpu_tbe_training_backward_gwd.so
  fbgemm_gpu_tbe_training_backward_pt2.so
  fbgemm_gpu_tbe_training_backward_vbe.so
  fbgemm_gpu_tbe_optimizers.so
  fbgemm_gpu_py.so
)

runner_preload_args=()
e2e_preload_args=()
if [[ -n "$LIBPYTHON" ]]; then
  runner_preload_args+=(--aott_library "$LIBPYTHON")
  e2e_preload_args+=(--aott_library "$LIBPYTHON")
fi
for lib in "${fbgemm_libs[@]}"; do
  path="$FBGEMM_ROOT/$lib"
  if [[ -f "$path" ]]; then
    runner_preload_args+=(--aott_library "$path")
    e2e_preload_args+=(--aott_library "$path")
  fi
done

python generative_recommenders/dlrm_v3/inference/end_to_end_test.py \
  --cpp_runner "$RUNNER" \
  --uih_max_seq_len 32 \
  --dense_backend aott \
  --keep_workdir \
  "${e2e_preload_args[@]}" \
  2>&1 | tee "$RUN_DIR/aott_generate.log"

AOTT_WORKDIR=$(grep -oE 'workdir: .*$' "$RUN_DIR/aott_generate.log" | tail -1 | sed 's/^workdir: //')
if [[ -z "$AOTT_WORKDIR" || ! -d "$AOTT_WORKDIR" ]]; then
  echo "failed to find AOT workdir" >&2
  exit 1
fi
printf '%s\n' "$AOTT_WORKDIR" > "$RUN_DIR/aott_workdir.txt"

find "$AOTT_WORKDIR" -maxdepth 1 -type f -name 'aott_*.so' | sort > "$RUN_DIR/generated_aott_libraries.txt"

generated_aott_args=()
while IFS= read -r lib; do
  [[ -n "$lib" ]] && generated_aott_args+=(--aott_library "$lib")
done < "$RUN_DIR/generated_aott_libraries.txt"

OUT="$RUN_DIR/preds_cpp_aott_nsys.pt"
{
  printf '%q ' "$RUNNER" "${runner_preload_args[@]}" "${generated_aott_args[@]}" \
    "$AOTT_WORKDIR/sparse.pt" "$AOTT_WORKDIR/dense.pt" "$AOTT_WORKDIR/inputs.pt" "$OUT"
  printf '\n'
} > "$RUN_DIR/runner_command.txt"

nsys profile \
  --force-overwrite=true \
  --sample=none \
  --trace=cuda,nvtx \
  --output="$RUN_DIR/hstu_runner_aott" \
  "$RUNNER" \
  "${runner_preload_args[@]}" \
  "${generated_aott_args[@]}" \
  "$AOTT_WORKDIR/sparse.pt" \
  "$AOTT_WORKDIR/dense.pt" \
  "$AOTT_WORKDIR/inputs.pt" \
  "$OUT" \
  2>&1 | tee "$RUN_DIR/runner_stdout_stderr.txt"

python - "$AOTT_WORKDIR" "$OUT" <<'PY' | tee "$RUN_DIR/parity.txt"
import sys
import torch

workdir, out = sys.argv[1], sys.argv[2]
eager = torch.load(f"{workdir}/preds_eager.pt", weights_only=False).to(torch.float32).cpu()
cpp = torch.load(out, weights_only=False).to(torch.float32).cpu()
print("eager", tuple(eager.shape), float(eager.sum()))
print("cpp", tuple(cpp.shape), float(cpp.sum()))
torch.testing.assert_close(eager, cpp, atol=1e-2, rtol=1e-2)
print("aott nsys C++ parity passed")
PY

nsys stats "$RUN_DIR/hstu_runner_aott.nsys-rep" > "$RUN_DIR/nsys_stats.txt" || true
nsys export --force-overwrite=true --type sqlite --output "$RUN_DIR/hstu_runner_aott.sqlite" "$RUN_DIR/hstu_runner_aott.nsys-rep" || true

find "$RUN_DIR" -maxdepth 1 -type f -printf '%p\t%s bytes\n' | sort
