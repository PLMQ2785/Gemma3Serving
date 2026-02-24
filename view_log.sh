#!/bin/bash

usage() {
    echo "vLLM Server Stop Script"
    echo "Usage: ./view_log.sh [GPU_ID]"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID    Target GPU index (0 or 1)"
    echo ""
    echo "Description:"
    echo "  Finds the process using the port associated with the GPU_ID and view log"
    exit 1
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

GPU_ID=$1
PORT=$((8030 + GPU_ID))
PID=$(lsof -t -i:$PORT)

if [ -z "$PID" ]; then
    echo "No vLLM process found on port $PORT (GPU $GPU_ID)."
else
    tail -f vllm_gpu$GPU_ID.log
fi
