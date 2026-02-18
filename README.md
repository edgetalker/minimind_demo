### Dataset
wget "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision=master&FilePath=sft_512.jsonl" \
     -O ./dataset/sft_512.jsonl
+ pretrain_hq.jsonl
+ sft-512.jsonl
### Train
+ Pretrain
![pretrain_768](./images/Pretrain-Epoch-4-BatchSize-32.png)
+ SFT
![full_sft_768](./images/Full-SFT-Epoch-2-BatchSize-16.png)