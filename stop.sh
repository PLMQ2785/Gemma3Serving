#!/bin/bash

usage() {
    echo "vLLM Server Stop Script"
    echo "Usage: ./stop.sh [GPU_ID]"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID    Target GPU index (0 or 1)"
    echo ""
    echo "Description:"
    echo "  Finds the process using the port associated with the GPU_ID and terminates it."
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
    echo "Stopping vLLM GPU $GPU_ID (PID: $PID)..."
    kill $PID
    while ps -p $PID > /dev/null; do sleep 1; done
    echo "Stopped successfully."
fi
