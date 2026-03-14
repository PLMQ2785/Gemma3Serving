uv run python -m vllm.entrypoints.openai.api_server \
       	--model ./exaone-4.0.1-32B-W4A16 \
       	--quantization compressed-tensors \
	--dtype bfloat16 \
	--max-model-len 4096  \
	--gpu-memory-utilization 0.9 \
	--trust-remote-code \
	--port 8030
