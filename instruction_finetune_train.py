from utils.dataset_loader import custom_collate_fn, InstructionDataset, format_input
from torch.utils.data import DataLoader
from utils.load_gpt2_weights import load_gpt2_weights_into_model
# from configs.config import GPT_CONFIG_355M
from utils.train_model import train_model
from model.gpt_model import GPTModel
from functools import partial
import tiktoken
import torch
import sys
import json
from pathlib import Path
from argparse import ArgumentParser


if __name__=="__main__":
    '''
    基于GPT2-355M权重的模型进行指令执行微调
    主要参数:
        --data_path(str):指令、输入、回复对数据集文件路径
        --config(str): 模型参数配置
        --gpt2_model_path(str): 加载GPT2-355M基础模型权重路径
        --model_path(str):微调后的模型权重保存路径
    '''

    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/instruction-data.json")
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355m.json")
    parser.add_argument("--gpt2_model_path", type=str, default="weights/pytorch_model_355m.bin")
    parser.add_argument("--model_path", type=str, default="weights/instruction_executor.pth")
    args = parser.parse_args()
    data_path, config, gpt2_model_path, model_path = vars(args).values()

    # 如果用于微调的原始数据文件不存在，则提示并退出程序
    if not Path(data_path).exists():
        print(f"用于指令微调的原始数据文件 {data_path} 不存在")
        sys.exit()
    
    # 如果用于微调的 GPT-2 模型权重文件不存在，则提示并退出程序
    if not Path(gpt2_model_path).exists():
        print(f"gpt2模型的权重文件 {gpt2_model_path} 不存在")
        sys.exit()

    # 加载配置文件
    with open(config) as f:
        cfg = json.load(f)


    ##############################
    # 训练初始化
    ##############################
    # 如果你有一台支持 CUDA 的 GPU 机器，那么大语言模型将自动在 GPU 上训练且不需要修改代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)


    ##############################
    # 准备指令微调数据集
    ##############################
    # 读取原始数据集
    with open(data_path, 'r') as file:
        data= json.load(file)

    # 划分数据集, 85%作为训练集, 10%作为测试集, 5%作为验证集
    train_portion = int(len(data) * 0.85) 
    test_protion = int(len(data) * 0.1) # 
    val_portion = len(data) - train_portion - test_protion 
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_protion]
    val_data = data[train_portion + test_protion:]

    #准备和装载数据集
    batch_size = 8 # 设置批次大小
    num_workers = 0 # 设置数据加载器的多线程工作线程数

    #为了custom_collate_fn函数应用于DataLoader类时重用所选择的device，利用partial预先填充设备参数
    custom_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    # 训练数据集
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size, # 每个批次的大小
        collate_fn = custom_collate_fn, # 自定义批次数据的整理方式
        shuffle = True, # 是否打乱数据顺序 防止模型过拟合
        drop_last = True, # 是否丢弃最后一个不完整的batch
        num_workers = num_workers # 多线程加载数据的工作线程数
    )

    # 验证数据集
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        collate_fn = custom_collate_fn,
        shuffle = True,
        drop_last = True,
        num_workers = num_workers
    )

    # 测试数据集
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = custom_collate_fn,
        shuffle = True,
        drop_last = True,
        num_workers = num_workers
    )


    ##############################
    # 初始化并加载带gpt2-255m权重的基础模型
    ##############################
    #实例化模型
    model = GPTModel(cfg)
    model.to(device)

    # 加载GPT2-355M基础模型的权重
    load_gpt2_weights_into_model(model, gpt2_model_path)


    ##############################
    # 训练模型以及保存指令微调权重
    ##############################
    # 初始化优化器，优化器是用于更新模型权重参数的算法，这里使用 AdamW 算法
    optimizer = torch.optim.AdamW(
        model.parameters(), # .parameters()方法返回模型的所有可训练权重参数
        lr=5e-4, # 学习率，即模型权重参数的梯度下降步长的系数，决定具体变化快慢，即 w = w - lr * dL/dw
        weight_decay=0.1 # 权重衰减，即模型权重参数的 L2 正则化系数
    )

    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    # 保存模型权重
    torch.save(model.state_dict(), model_path)