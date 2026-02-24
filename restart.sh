#!/bin/bash

usage() {
    echo "vLLM Server Restart Script"
    echo "Usage: ./restart.sh [GPU_ID]"
    echo ""
    echo "Arguments:"
    echo "  GPU_ID    Target GPU index (0 or 1)"
    exit 1
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

GPU_ID=$1

echo "Restarting vLLM on GPU $GPU_ID..."
./stop.sh $GPU_ID
sleep 2
./start.sh $GPU_ID
