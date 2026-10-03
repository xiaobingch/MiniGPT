from  utils.dataset_loader import create_dataloader_v1
# from configs.config import GPT_CONFIG_124M
from utils.train_model import train_model
from model.gpt_model import GPTModel
import tiktoken
import torch
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

if __name__=="__main__":
    '''
    初始化大模型并进行模型预训练

    Args:
        --data_path(str):原始数据集文件路径
        --configP(str): 模型参数配置
        --model_path(str):模型权重保存路径
    '''

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_124m_ctx256.json")
    parser.add_argument("--data_path",type=str, default="data/the-verdict.txt")
    parser.add_argument("--model_path",type=str, default="weights/model_ctx256.pth")

    args = parser.parse_args()
    config, data_path, model_path = args.config, args.data_path, args.model_path

    if not Path(data_path).exists():
        print(f"用于训练的数据{data_path}不存在")
        sys.exit(1)

    with open(config) as f:
        cfg = json.load(f)

    ##############################
    # 初始化
    ##############################
    # 如果你有一台支持 CUDA 的 GPU 机器，那么大语言模型将自动在 GPU 上训练且不需要修改代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)


    ##############################
    # 准备数据集
    ##############################
    # 读取原始数据集
    with open(data_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # 切分训练集和验证集，这里将 90% 的数据作为训练集，10% 的数据作为验证集
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    batch_size = 2      # 设置批次大小
    num_workers = 0     # 设置数据加载器的多线程工作线程数

    # 准备训练数据集
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )

    # 准备验证数据集
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )


    ##############################
    # 初始化模型
    ##############################
    model = GPTModel(cfg)
    model.to(device)


    ##############################
    # 训练模型并保存权重
    ##############################
    # 初始化优化器，优化器是用于更新模型权重参数的算法，这里使用 AdamW 算法
    optimizer = torch.optim.AdamW(
        model.parameters(), # .parameters()方法返回模型的所有可训练权重参数
        lr=0.0004, # 学习率，即模型权重参数的梯度下降步长的系数，决定具体变化快慢，即 w = w - lr * dL/dw
        weight_decay=0.1 # 权重衰减，即模型权重参数的 L2 正则化系数
    )

    # 训练模型
    train_losses, val_losses, tokens_seen = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device,
        num_epochs=10, 
        eval_freq=5, 
        eval_iter=5,
        start_context="all I think", 
        tokenizer=tokenizer
    )

    # 保存模型权重
    torch.save(model.state_dict(), model_path)