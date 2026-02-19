CUDA_VISIBLE_DEVICES=1 uv run python -m vllm.entrypoints.openai.api_server \
       	--model ./gemma-3-27b-it-NVFP4 \
       	--quantization compressed-tensors \
	--dtype bfloat16 \
	--max-model-len 4096  \
	--gpu-memory-utilization 0.9 \
	--trust-remote-code \
	--port 8030
