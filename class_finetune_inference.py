from model.gpt_model import GPTModel
from configs.config import GPT_CONFIG_124M
from utils.model_inference import classify_review
import torch
import tiktoken
import sys
import json
from pathlib import Path
from argparse import ArgumentParser


if __name__=="__main__":

    """
    垃圾消息分类

    Args:
        --config (str): 模型配置参数文件路径
        --model_path (str): 微调后保存模型权重文件路径
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_124m.json")
    parser.add_argument("--model_path", type=str, default="weights/review_classifier.pth")
    args = parser.parse_args()
    config, model_path = vars(args).values()

    # 如果模型权重文件不存在，则提示并退出程序
    if not Path(model_path).exists():
        print(f"模型权重文件 {model_path} 不存在，请先进行模型份分类微调")
        sys.exit()

    with open(config) as f:
        cfg = json.load(f)


    ##############################
    # 初始化
    ##############################
    # 如果你有一台支持 CUDA 的 GPU 机器，那么大语言模型将自动在 GPU 上训练且不需要修改代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")


    ##############################
    # 初始化模型
    ##############################
    model = GPTModel(cfg)
    model.to(device)


    ##############################
    # 修改模型适应于分类任务
    # 并加载 GPT-2 预训练权重
    ##############################
    # 评论分类任务，因此我们只需替换最后的输出层即可，
    # 该层原本是将输入映射为 50257 维的向量，即词汇表的大小，
    # 现在将其输出层作用改为映射为 2 维的向量，即 0/1 两类的分类器
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features = GPT_CONFIG_124M['emb_dim'],
        out_features = num_classes
    )

    # 加载之前训练好的模型权重参数，weights_only=True 表示只加载模型参数，不加载优化器等状态信息
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 切换为推理模式，将禁用 dropout 等只在训练时使用的功能
    model.eval()


    ##############################
    # 使用模型进行消息分类
    ##############################
    print("开始对话（输入'exit'退出）\n")
    while True:
        input_text = input("User: ")
        if input_text.lower() == '':
            print("输入不能为空！")
            continue
        if input_text.lower() == 'exit':
            break

        label = classify_review(
            text=input_text, 
            model=model, 
            tokenizer=tokenizer,
            device=device,
            max_length=cfg["context_length"]
        )

        print(f"模型: {label}\n")