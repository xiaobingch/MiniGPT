from model.gpt_model import GPTModel
from configs.config import GPT_CONFIG_355M
from utils.model_inference import generate, token_ids_to_text, text_to_token_ids
from utils.dataset_loader import format_input
import torch
import tiktoken


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
model = GPTModel(GPT_CONFIG_355M)
model.to(device)

# 加载指令微调后的模型权重（基于GPT2-355M预训练权重）
model_path = 'weights/instruction_executor.pth' 
model.load_state_dict(
    torch.load(
        model_path, 
        map_location=device, # 将GPU权重中的张量统一映射到 CPU
        weights_only=True # weights_only=True 只加载模型参数，不加载优化器等状态信息
    )
) 

# 切换为推理模式，将禁用 dropout 等只在训练时使用的功能
model.eval()


##############################
# 指令推理
##############################
instruction_text = "Name the author of 'Pride and Prejudice'."
input_text = ""

format_input = format_input({
    "instruction": instruction_text,
    "input": input_text
})

print(f"User:{instruction_text}")

token_ids = generate(
    model=model,
    idx=text_to_token_ids(format_input, tokenizer).to(device),
    max_new_tokens=256,
    context_size=GPT_CONFIG_355M['context_length'],
    eos_id=50256
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(format_input):].replace("### Response:", "").strip()

print(f"\nModel:{response_text}")

