#!/bin/bash

usage() {
    echo "vLLM Server Start Script"
    echo "Usage: ./start.sh [GPU_ID]"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID    Target GPU index (0 or 1)"
    echo ""
    echo "Options:"
    echo "  -h, --help  Show this help message"
    echo ""
    echo "Description:"
    echo "  Starts the vLLM server on the specified GPU."
    echo "  GPU 0 uses Port 8030, GPU 1 uses Port 8031."
    exit 1
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

GPU_ID=$1
PORT=$((8030 + GPU_ID))
LOG_FILE="vllm_gpu${GPU_ID}.log"

if lsof -i:$PORT > /dev/null; then
    echo "Error: Port $PORT (GPU $GPU_ID) is already in use."
    exit 1
fi

echo "Starting vLLM on GPU $GPU_ID (Port: $PORT)..."

CUDA_VISIBLE_DEVICES=$GPU_ID nohup uv run python -m vllm.entrypoints.openai.api_server \
        --model ./gemma-3-27b-it-NVFP4 \
        --quantization compressed-tensors \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --port $PORT > $LOG_FILE 2>&1 &

echo "vLLM GPU $GPU_ID is running in background."
echo "Log file: $LOG_FILE"
