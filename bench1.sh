#!/bin/bash
set -e

# ============================================
# lm-eval 벤치마크
# Usage: ./scripts/bench_lmeval.sh <model_name> [port]
# ============================================

MODEL_NAME=${1:?"❌ Usage: $0 <model_name> <model_path> [port]"}
MODEL_PATH=${2:?"❌ Usage: $0 <model_name> <model_path> [port]"}
PORT=${3:-8001}
BASE_URL="http://localhost:${PORT}/v1/completions"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/lm-eval/${MODEL_NAME}/${TIMESTAMP}"

mkdir -p "${RESULT_DIR}"

#cd "$(dirname "$0")/../lm-eval"

echo "=========================================="
echo "📊 lm-eval 벤치마크"
echo "   모델: ${MODEL_NAME}"
echo "   결과: ${RESULT_DIR}"
echo "=========================================="

# 공통 인자
COMMON=(
    --model local-completions
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=4,max_retries=3,tokenized_requests=False,tokenizer=${MODEL_PATH}"
    --batch_size auto
)

# ---------- 벤치마크 정의 ----------
# (태스크명, fewshot, 출력 폴더명)
declare -a BENCHMARKS=(
    "mmlu,arc_challenge,hellaswag,winogrande|5|general"
    "gsm8k|5|math"
    "kmmlu|5|korean"
    "humaneval|0|code"
)

TOTAL=${#BENCHMARKS[@]}
CURRENT=0

for BENCH in "${BENCHMARKS[@]}"; do
    IFS='|' read -r TASKS FEWSHOT LABEL <<< "${BENCH}"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "🔷 [${CURRENT}/${TOTAL}] ${LABEL}: ${TASKS} (${FEWSHOT}-shot)"
    echo "------------------------------------------"

    uv run lm_eval "${COMMON[@]}" \
        --tasks "${TASKS}" \
        --num_fewshot "${FEWSHOT}" \
        --output_path "${RESULT_DIR}/${LABEL}/" \
        2>&1 | tee "${RESULT_DIR}/${LABEL}.log"

    echo "✅ ${LABEL} 완료!"
done

echo ""
echo "=========================================="
echo "✅ lm-eval 전체 완료: ${RESULT_DIR}"
echo "=========================================="

