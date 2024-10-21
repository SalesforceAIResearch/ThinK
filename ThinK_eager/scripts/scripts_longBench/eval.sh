export CUDA_VISIBLE_DEVICES=4


model_path="meta-llama/Meta-Llama-3-8B-Instruct"
method="SnapKV" # Support SnapKV, H2O
max_capacity_prompts=512
save_dir="test" # path to result save_dir
pruning_ratio=0.4
recent_size=32

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --save_dir ${save_dir} \
    --use_cache True \
    --pruning_ratio ${pruning_ratio} \
    --recent_size ${recent_size}
