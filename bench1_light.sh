#!/bin/bash
set -e

# ============================================
# lm-eval Light 벤치마크 (Full 항목, 문항 수 제한)
# Usage: ./bench_light.sh <model_name> <model_path> [port] [limit]
# ============================================

MODEL_NAME=${1:?"❌ Usage: $0 <model_name> <model_path> [port] [limit]"}
MODEL_PATH=${2:?"❌ Usage: $0 <model_name> <model_path> [port] [limit]"}
PORT=${3:-8030}
LIMIT=${4:-100}

BASE_URL="http://localhost:${PORT}/v1/completions"
CHAT_URL="http://localhost:${PORT}/v1/chat/completions"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/lm-eval/${MODEL_NAME}/light_${LIMIT}_${TIMESTAMP}"

mkdir -p "${RESULT_DIR}"

echo "=========================================="
echo "⚡ lm-eval LIGHT 벤치마크"
echo "   모델:       ${MODEL_NAME}"
echo "   토크나이저: ${MODEL_PATH}"
echo "   포트:       ${PORT}"
echo "   샘플 제한:  ${LIMIT}개/서브태스크"
echo "   결과:       ${RESULT_DIR}"
echo "=========================================="

# ---- loglikelihood 태스크 (local-completions) ----
COMMON_LL=(
    --model local-completions
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=4,max_retries=3,tokenized_requests=False,tokenizer=${MODEL_PATH}"
    --batch_size auto
    --limit ${LIMIT}
)

# ---- generation 태스크 (local-chat-completions) ----
COMMON_GEN=(
    --model local-chat-completions
    --model_args "model=${MODEL_NAME},base_url=${CHAT_URL},num_concurrent=4,max_retries=3,tokenized_requests=False"
    --batch_size auto
    --limit ${LIMIT}
)

# ---------- 벤치마크 정의 ----------
# 형식: 태스크|fewshot|라벨|타입(ll=loglikelihood, gen=generation)
declare -a BENCHMARKS=(
    "mmlu,arc_challenge,hellaswag,winogrande|5|general|ll"
    "gsm8k|5|math|gen"
    "kmmlu|5|korean|ll"
    "humaneval|0|code|gen"
)

TOTAL=${#BENCHMARKS[@]}
CURRENT=0
START_TIME=$(date +%s)

for BENCH in "${BENCHMARKS[@]}"; do
    IFS='|' read -r TASKS FEWSHOT LABEL TYPE <<< "${BENCH}"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "🔷 [${CURRENT}/${TOTAL}] ${LABEL}: ${TASKS} (${FEWSHOT}-shot, limit=${LIMIT})"
    echo "------------------------------------------"

    TASK_START=$(date +%s)

    if [ "${TYPE}" = "gen" ]; then
        uv run lm_eval "${COMMON_GEN[@]}" \
            --tasks "${TASKS}" \
            --num_fewshot "${FEWSHOT}" \
            --output_path "${RESULT_DIR}/${LABEL}/" \
            2>&1 | tee "${RESULT_DIR}/${LABEL}.log"
    else
        uv run lm_eval "${COMMON_LL[@]}" \
            --tasks "${TASKS}" \
            --num_fewshot "${FEWSHOT}" \
            --output_path "${RESULT_DIR}/${LABEL}/" \
            2>&1 | tee "${RESULT_DIR}/${LABEL}.log"
    fi

    TASK_END=$(date +%s)
    TASK_ELAPSED=$(( TASK_END - TASK_START ))
    TASK_MIN=$(( TASK_ELAPSED / 60 ))
    echo "✅ ${LABEL} 완료! (${TASK_MIN}분 ${TASK_ELAPSED}초)"
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo ""
echo "=========================================="
echo "✅ Light 벤치마크 전체 완료!"
echo "   총 소요 시간: ${TOTAL_MIN}분 (${TOTAL_ELAPSED}초)"
echo "   결과: ${RESULT_DIR}"
echo "=========================================="

# ---- 결과 요약 출력 ----
echo ""
echo "📋 결과 요약:"
echo "------------------------------------------"
for DIR in "${RESULT_DIR}"/*/; do
    if [ -d "$DIR" ]; then
        LABEL=$(basename "$DIR")
        # results.json에서 점수 추출
        RESULTS_FILE=$(find "$DIR" -name "results.json" 2>/dev/null | head -1)
        if [ -n "$RESULTS_FILE" ]; then
            echo "📊 ${LABEL}:"
            python3 -c "
import json, sys
with open('${RESULTS_FILE}') as f:
    data = json.load(f)
results = data.get('results', {})
for task, metrics in sorted(results.items()):
    acc = metrics.get('acc,none', metrics.get('acc_norm,none', metrics.get('exact_match,none', 'N/A')))
    if acc != 'N/A':
        acc = f'{float(acc)*100:.1f}%'
    print(f'  {task}: {acc}')
" 2>/dev/null || echo "  (결과 파싱 실패)"
        fi
    fi
done

