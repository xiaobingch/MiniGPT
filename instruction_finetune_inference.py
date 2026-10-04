from model.gpt_model import GPTModel
# from configs.config import GPT_CONFIG_355M
from utils.model_inference import generate, token_ids_to_text, text_to_token_ids
from utils.dataset_loader import format_input
import torch
import tiktoken
import sys
import json
from pathlib import Path
from argparse import ArgumentParser


if __name__=="__main__":
    '''
    指令执行应用
    Args:
        --config(str): gpt2-355m配置文件
        --model_path(str): 基于gpt2-355m基础权重指令微调的后的权重
        --max_new_tokens(int): 新生成的 token 最大数量
        --temperature(float): 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定.例如25
        --top_k(int): top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定.例如1.4
    '''
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355m.json")
    parser.add_argument("--model_path", type=str, default="weights/instruction_executor.pth")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()
    config, model_path, max_new_tokens, temperature, top_k = vars(args).values()

    # 如果模型权重文件不存在，则提示并退出程序
    if not Path(model_path).exists():
        print(f"模型权重文件 {model_path} 不存在，请先进行模型指令微调")
        sys.exit()

    # 解析json 配置文件
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
    # 初始化模型以及加载指令微调后的权重
    ##############################
    model = GPTModel(cfg)
    model.to(device)

    # 加载指令微调后的模型权重（基于GPT2-355M预训练权重）
    model.load_state_dict(
        torch.load(
            model_path, 
            map_location=device, # 将GPU权重中的张量统一映射到 CPU或GPU
            weights_only=True # weights_only=True 只加载模型参数，不加载优化器等状态信息
        )
    ) 

    # 切换为推理模式，将禁用 dropout 等只在训练时使用的功能
    model.eval()


    ##############################
    # 指令推理,交互式对话
    ##############################
    print("开始对话（输入'exit'退出）\n")
    while True:
        # 指令
        instruction_text = input("任务指令: ")
        if instruction_text.lower() == '':
            print('指令不能为空！')
            continue
        if instruction_text.lower() == 'exit':
            break

        # 输入
        input_text = input("任务输入: ")
        if input_text.lower() == 'exit':
            break
        if input_text.lower() == '':
            input_text = ""

        # 格式化input，载入指令模版
        formatted_input = format_input({
            "instruction": instruction_text,
            "input": input_text
        })

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(formatted_input, tokenizer).to(device),
            max_new_tokens=max_new_tokens,
            context_size=cfg['context_length'],
            temperature=temperature,
            top_k=top_k,
            eos_id=50256
        )
        
        # 词元id转词元，删除响应内容中的用户输入部分以及Response模版
        response_text = token_ids_to_text(token_ids, tokenizer)[len(formatted_input):].replace("### Response:", "").strip()

        print(f"模型: {response_text}\n")