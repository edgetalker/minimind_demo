#!/bin/bash
echo "========== 4卡RTX4090 Pretrain训练 =========="
echo "模型: MiniMind2 (768维, 16层)"
echo "数据: pretrain_hq.jsonl"
echo "监控: SwanLab实时可视化"
echo ""

torchrun --nproc_per_node=4 --master_port=29500 ./trainer/train_pretrain.py \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --data_path ./dataset/pretrain_hq.jsonl \
    --max_seq_len 320 \
    --batch_size 32 \
    --accumulation_steps 16 \
    --learning_rate 1.5e-4 \
    --epochs 4 \
    --save_dir ../checkpoints \
    --save_weight pretrain_768 \
    --save_interval 1000 \
    --log_interval 200 \
    --dtype bfloat16 \
    --num_workers 8 \
    --grad_clip 1.0 \
    --use_moe 0 \
    --from_weight pretrain_768 \
    --from_resume 1 \
    --use_wandb \
    --wandb_project "MiniMind2-Pretrain"

echo ""
echo "✅ 训练完成！"
