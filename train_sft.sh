#!/bin/bash
# 一键启动/续训脚本

SESSION_NAME="minimind"

# 如果会话已存在，连接到它
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "✅ 训练会话已存在，正在连接..."
    tmux attach -t $SESSION_NAME
    exit 0
fi

# 创建新会话并启动训练
echo "🚀 启动新的训练会话..."
tmux new-session -d -s $SESSION_NAME

# 在tmux中执行训练命令
tmux send-keys -t $SESSION_NAME 
torchrun --nproc_per_node=4 --master_port=29500 ./trainer/train_full_sft.py \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --data_path ./dataset/sft_512.jsonl \
    --max_seq_len 350 \
    --batch_size 16 \
    --accumulation_steps 8 \
    --learning_rate 1e-5 \
    --epochs 2 \
    --save_dir ../out \
    --save_weight full_sft_768 \
    --save_interval 500 \
    --log_interval 100 \
    --dtype bfloat16 \
    --num_workers 8 \
    --grad_clip 1.0 \
    --use_moe 0 \
    --from_weight pretrain_768 \
    --from_resume 1 \
    --use_wandb \
    --wandb_project 'MiniMind2-SFT'
" C-m

echo "✅ 训练已启动！"
echo ""
echo "连接到训练会话: tmux attach -t $SESSION_NAME"
echo "分离会话: Ctrl+B 然后按 D"
echo "查看所有会话: tmux ls"
echo ""

# 自动连接
sleep 2
tmux attach -t $SESSION_NAME