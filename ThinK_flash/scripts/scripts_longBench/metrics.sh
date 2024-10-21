
model_path="meta-llama-3-8b-instruct_SnapKV_max_capacity_prompts_512_"
results_dir="results_long_bench"

python3 eval.py \
    --model_name ${model_path} \
    --results_dir ${results_dir}
