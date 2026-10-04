from model.gpt_model import GPTModel
# from configs.config import GPT_CONFIG_124M
from utils.model_inference import generate, token_ids_to_text, text_to_token_ids
from utils.load_gpt2_weights import load_gpt2_weights_into_model
import torch
import tiktoken
import sys
import json
from pathlib import Path
from argparse import ArgumentParser


if __name__=="__main__":
    '''
    基础模型文本补全

    Args:
        --config(str): 模型参数配置文件
        --model_path(str):模型权重加载路径
        --max_new_tokens(int): 新生成的 token 最大数量
        --temperature(float): 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定.例如25
        --top_k(int):top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定.例如1.4
    '''
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_124m_ctx256.json")
    parser.add_argument("--model_path", type=str, default="weights/model_ctx256.pth")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    config, model_path, max_new_tokens, temperature, top_k = vars(args).values()

    # 如果模型权重文件不存在，则提示并退出程序
    if not Path(model_path).exists():
        print(f"模型权重文件 {model_path} 不存在，请先训练模型")
        sys.exit()

    with open(config) as f:
        cfg = json.load(f)

    ##############################
    # 初始化
    ##############################
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 如果你有一台支持 CUDA 的 GPU 机器，那么大语言模型将自动在 GPU 上训练且不需要修改代码
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)


    ##############################
    # 初始化基础模型以及加载加载基础权重
    ##############################
    model = GPTModel(cfg)
    model.to(device)

    # 加载模型权重
    if model_path.endswith("pytorch_model.bin"):
        # 如果权重文件名为 pytorch_model.bin，表明是加载了 GPT-2 预训练权重，
        # 需要使用 load_gpt2_weights_into_model 将 GPT-2 模型权重加载到自定义模型中
        load_gpt2_weights_into_model(model, model_path)
    else:
        # 加载之前训练好的模型权重参数，weights_only=True 表示只加载模型参数，不加载优化器等状态信息
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # 切换为推理模式，将禁用 dropout 等只在训练时使用的功能
    model.eval()


    ##############################
    # 基础预训练模型推理
    ##############################
    print("开始对话（输入'exit'退出）\n")
    while True:
        input_text = input("用户: ")
        if input_text.lower() == '':
            print("输入不能为空！")
            continue
        if input_text.lower() == 'exit':
            break
        
        # 文本生成
        encode = text_to_token_ids(input_text, tokenizer).to(device)
        token_ids = generate(
            model=model, 
            idx=encode, 
            max_new_tokens=max_new_tokens, 
            context_size=cfg['context_length'],
            temperature=temperature,
            top_k=top_k
        )
        output_text = token_ids_to_text(token_ids, tokenizer)
        print(f"\n模型: {output_text}\n")