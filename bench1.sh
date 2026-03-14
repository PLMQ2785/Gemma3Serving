uv run lm_eval --model local-completions \
    --model_args model=gemma-3-27b-it-W4A16,base_url=http://localhost:8030/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False \
    --tasks mmlu,arc_challenge,hellaswag,gsm8k,winogrande \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path ./results/gemma-3-27b-w4a16/
