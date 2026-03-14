uv run lm_eval --model local-completions \
    --model_args model=exaone-4.0.1-32B-W4A16,base_url=http://localhost:8030/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False \
    --tasks kmmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path ./results/exaone-4.0.1-32B-W4A16/kmmlu/
