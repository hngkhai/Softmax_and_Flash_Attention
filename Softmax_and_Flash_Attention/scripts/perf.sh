#!/bin/bash
#SBATCH -o ./Project4-profiling-results.txt
#SBATCH -p Release
#SBATCH -J Project4-Profiling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --exclusive

set -euo pipefail

echo "---- tool check ----"
which perf || true
which nsys || true
nsys --version || true
perf --version || true
echo "--------------------"


echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"(not set)"}"
echo "------------------------------------------------------------------"

# =============== 可选：让 triton cache 不写到 home（避免 NFS 慢/权限问题） ===============
mkdir -p triton_cache
export TRITON_CACHE_DIR="$PWD/triton_cache"

# =============== 输出目录 ===============
mkdir -p perf_data nsys_data

# =============== perf 推荐参数（对 python 尤其有用） ===============
# -g 需要 call graph；--call-graph dwarf 更稳（但开销更大）
PERF_RECORD_ARGS="record -g --call-graph dwarf -F 999"

# =============== helper：统一跑 perf stat / perf record / nsys ===============
run_perf_and_nsys () {
  local tag="$1"
  shift
  local cmd=( "$@" )

  echo "=================================================="
  echo "Profiling: ${tag}"
  echo "CMD: ${cmd[*]}"
  echo "=================================================="

  # 1) perf stat
  srun -n 1 --cpus-per-task 1 --gres=gpu:1 \
    perf stat -e cpu-cycles,instructions,cache-misses,page-faults \
    -o "perf_data/perfstat_${tag}.txt" \
    -- "${cmd[@]}"

  # 2) perf record
  srun -n 1 --cpus-per-task 1 --gres=gpu:1 \
    perf ${PERF_RECORD_ARGS} \
    -o "perf_data/perf_${tag}.data" \
    -- "${cmd[@]}"

  # 3) nsys
  if command -v nsys >/dev/null 2>&1; then
    echo "[nsys] start..."
    
    # FIX: Manually remove old files instead of using --force-overwrite
    rm -f "nsys_data/nsys_${tag}.qdrep" "nsys_data/nsys_${tag}.sqlite"

    srun -n 1 --cpus-per-task 1 --gres=gpu:1 \
      nsys profile \
        -t cuda,nvtx,osrt \
        -o "nsys_data/nsys_${tag}" \
        -- "${cmd[@]}" \
      > "nsys_data/nsys_${tag}.log" 2>&1 \
      || echo "[nsys] FAILED, see nsys_data/nsys_${tag}.log"

    echo "[nsys] done."
  else
    echo "[nsys] not found on node. Skip."
  fi
}

# ====================== Part 1: Triton Softmax ======================
echo ">>> Part 1: Triton Softmax (python)"
run_perf_and_nsys "part1_triton_softmax" \
  python3 ./part1_softmax_and_vector_add/triton_softmax.py

# ====================== Part 2A: CUDA Softmax (compile + run) ======================
echo ">>> Part 2: CUDA Softmax (nvcc compile)"
nvcc ./part1_softmax_and_vector_add/cuda_softmax.cu -o ./part1_softmax_and_vector_add/cuda_softmax_exec
echo "Compilation successful."
echo ""

N=8192
C=4096

echo ">>> Part 2: CUDA Softmax (exec)"
run_perf_and_nsys "part2_cuda_softmax_N${N}_C${C}" \
  ./part1_softmax_and_vector_add/cuda_softmax_exec "${N}" "${C}"

# ====================== Part 2B: Triton Flash Attention ======================
echo ">>> Part 2: Triton Flash Attention (python)"
run_perf_and_nsys "part2_triton_flash_attention" \
  python3 ./part2_flash_attention/triton_part.py

# ====================== Part 3: Triton Sparse Flash Attention ======================
echo ">>> Part 3: Triton Sparse Flash Attention (python)"
run_perf_and_nsys "part3_triton_sparse_flash_attention" \
  python3 ./part3_sparse_flash_attention/triton_part.py

echo "------------------------------------------------------------------"
echo "All profiling finished!"
echo "Job finished at: $(date)"
