#usr/bin/env bash 

# ============= 训练参数配置区 =============
# 路径配置
model_path="/work/mount/presetdata/openmind/qwen1.5_7b"
train_data="/work/cache/train-SFT/dataset/peft_train_strict/peft_train_strict_reasoning.jsonl"
eval_data="/work/cache/train-SFT/dataset/mixed/eval_strict.parquet"
output_dir="./output/Qwen1_5"
log_file="./output/finetune_qwen1.5_7b.log"

# 分布式训练配置
num_gpus=8
master_port=27501

# 训练超参数
batch_size=8
eval_batch_size=32
grad_accum_steps=8
num_epochs=4
learning_rate=5e-6
weight_decay=0.01
warmup_ratio=0.05
warmup_steps=100
max_length=1024
save_steps=50
save_total_limit=20
eval_steps=25
eval_samples=500
seed=1234

# LoRA配置
use_peft="False"
peft_method="lora"
lora_r=16
lora_alpha=32
lora_dropout=0.05

# ============= 环境配置区 =============
# 激活环境
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate openmind_finetune

# NPU配置
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12

# 创建输出目录
if [ -d ./output ];then
    rm -rf ./output
    mkdir -p ./output
else
    mkdir -p ./output
fi

# 创建日志文件并写入训练参数
cat << EOF > ${log_file}
=============== Training Parameters ===============
Model path: ${model_path}
Training data: ${train_data}
Evaluation data: ${eval_data}
Output directory: ${output_dir}

Distributed Training:
- Number of GPUs: ${num_gpus}
- Master port: ${master_port}

Training Hyperparameters:
- Batch size per device: ${batch_size}
- Eval batch size: ${eval_batch_size}
- Gradient accumulation steps: ${grad_accum_steps}
- Number of epochs: ${num_epochs}
- Learning rate: ${learning_rate}
- Weight decay: ${weight_decay}
- Warmup ratio: ${warmup_ratio}
- Warmup steps: ${warmup_steps}
- Model max length: ${max_length}
- Save steps: ${save_steps}
- Save total limit: ${save_total_limit}
- Evaluation steps: ${eval_steps}
- Evaluation samples: ${eval_samples}
- Random seed: ${seed}

LoRA Configuration:
- Use PEFT: ${use_peft}
- PEFT method: ${peft_method}
- LoRA rank: ${lora_r}
- LoRA alpha: ${lora_alpha}
- LoRA dropout: ${lora_dropout}
================================================

EOF

echo "start finetune..."
# 启动训练
torchrun --nproc_per_node=${num_gpus} --master_port=${master_port} train.py \
    --model_name_or_path ${model_path} \
    --train_data_path ${train_data} \
    --deepspeed ds_config.json \
    --bf16 True \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --warmup_steps ${warmup_steps} \
    --lr_scheduler_type "cosine" \
    --use_peft ${use_peft} \
    --peft_method ${peft_method} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --model_max_length ${max_length} \
    --seed ${seed} \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --eval_data_path ${eval_data} \
    --metric_for_best_model "eval_accuracy" \
    --greater_is_better True \
    --eval_steps ${eval_steps} \
    --per_device_eval_batch_size ${eval_batch_size} \
    --eval_num_samples ${eval_samples} \
    --load_best_model_at_end False \
    >> ${log_file} 2>&1 &
wait