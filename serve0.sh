uv run python -m vllm.entrypoints.openai.api_server \
       	--model ./gemma-3-27b-it-W4A16 \
	--served-model-name gemma3-27b \
       	--quantization compressed-tensors \
	--dtype bfloat16 \
	--max-model-len 4096  \
	--gpu-memory-utilization 0.9 \
	--trust-remote-code \
	--port 8030
