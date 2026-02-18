#!/usr/bin/env python3
# upload_to_hf.py - 上传模型到Hugging Face

from huggingface_hub import HfApi, create_repo
import os

# ===== 配置区域 =====
USERNAME = "swagger00"
REPO_NAME = "minimind-demo"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

# 要上传的文件
FILES_TO_UPLOAD = [
    {
        "local_path": "checkpoints/full_sft_768.pth",
        "repo_path": "full_sft_768.pth",
        "description": "SFT-epoch-2模型权重"
    },
    {
        "local_path": "checkpoints/full_sft_768_resume.pth", 
        "repo_path": "full_sft_768_resume.pth",
        "description": "训练checkpoint（包含优化器状态）"
    },
    {
        "local_path": "checkpoints/pretrain_768.pth",
        "repo_path": "pretrain_768.pth",
        "description": "Pretrain-epoch-4模型权重"
    },
    {
        "local_path": "checkpoints/pretrain_768_resume.pth", 
        "repo_path": "pretrian_768_resume.pth",
        "description": "训练checkpoint（包含优化器状态）"
    }
]

# ===== 主程序 =====
def main():
    print("=" * 50)
    print("  上传 MiniMind2 模型到 Hugging Face")
    print("=" * 50)
    print()
    
    api = HfApi()
    
    # 1. 创建仓库
    print(f"📦 创建仓库: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,  # 如果已存在则跳过
            private=False   # 改为True则创建私有仓库
        )
        print(f"✅ 仓库创建成功！")
        print(f"🔗 访问地址: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"⚠️  仓库可能已存在: {e}")
    
    print()
    
    # 2. 上传文件
    for file_info in FILES_TO_UPLOAD:
        local_path = file_info["local_path"]
        repo_path = file_info["repo_path"]
        
        if not os.path.exists(local_path):
            print(f"❌ 文件不存在: {local_path}")
            continue
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"📤 上传: {local_path} ({file_size:.1f} MB)")
        print(f"   → {repo_path}")
        
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"✅ 上传成功: {file_info['description']}")
        except Exception as e:
            print(f"❌ 上传失败: {e}")
        
        print()
    
    # 3. 上传README
    print("📝 创建README...")
    readme_content = f"""---
language:
- zh
tags:
- MiniMind
- SFT
- Chinese
license: apache-2.0
---

# MiniMind2

这是一个基于 MiniMind 架构训练的demo模型。

## 模型信息

- **模型大小**: 768维 × 16层 ≈ 104M 参数
- **训练数据**: Pretrian数据(~1.9GB) + SFT数据集 (~7.5GB)
- **训练轮数**: 4 epochs + 2 epochs
- **最终Loss**: ~2.5
- **训练时长**: ~16小时 (4×GPU)

## 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `pretrain_768.pth` | ~217MB | 预训练模型权重 |
| `pretrain_768_resume.pth` | ~1.0GB | 训练checkpoint（续训使用） |
| `full_sft_768.pth` | ~217MB | 最终模型权重（推理使用） |
| `full_sft_768_resume.pth` | ~1.0GB | 训练checkpoint（续训使用） |


## 使用方法

### 1. 下载模型
```python
from huggingface_hub import hf_hub_download

# 下载推理权重
model_path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="full_sft_768.pth"
)

# 或下载checkpoint（如需续训）
checkpoint_path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="full_sft_768_resume.pth"
)
```

### 2. 加载模型
```python
import torch
from model.model_minimind import MiniMind  # 需要MiniMind代码

# 加载模型
model = MiniMind(...)
model.load_state_dict(torch.load(model_path))
model.eval()

# 推理
output = model.generate("你好")
```

## 训练配置
```yaml
模型配置:
  hidden_size: 768
  num_hidden_layers: 16
  
训练超参数:
  batch_size: 16
  accumulation_steps: 8
  learning_rate: 1e-5
  epochs: 2
  dtype: bfloat16
```

## 项目链接

- GitHub: https://github.com/edgetalker/minimind_demo
- 原始项目: [MiniMind](https://github.com/jingyaogong/minimind)

## License

Apache 2.0
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("✅ README已创建")
    except Exception as e:
        print(f"⚠️  README创建失败: {e}")
    
    print()
    print("=" * 50)
    print("🎉 上传完成！")
    print(f"🔗 访问你的模型: https://huggingface.co/{REPO_ID}")
    print("=" * 50)

if __name__ == "__main__":
    main()