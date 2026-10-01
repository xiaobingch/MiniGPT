import json
import os
import urllib.request

##############################
# 7.2 准备数据集
##############################

# 下载数据集
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    with open(file_path, 'r') as file:
        data= json.load(file)
    return data

file_path = "../data/instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)
# print(len(data))
# print(data[50])
# 1100
# {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}

# print(data[999])
# {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."} 空input

# 实现提示词格式函数
def format_input(entry):
    instruciton_text = (
        f"Below is an instruction that describes a task."
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    )
    return instruciton_text + input_text

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
# print(model_input + desired_response)
# Below is an instruction that describes a task.Write a response that appropriately completes the request.

# ### Instruction:
# Identify the correct spelling of the following word.

# ### Input:
# Ocassion

# ### Response:
# The correct spelling is 'Occasion.'

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
# print(model_input + desired_response)
# Below is an instruction that describes a task.Write a response that appropriately completes the request.

# ### Instruction:
# What is an antonym of 'complicated'?

# ### Response:
# An antonym of 'complicated' is 'simple'.

# 划分数据集
train_portion = int(len(data) * 0.85) # 85%作为训练集
test_protion = int(len(data) * 0.1) # 10%作为测试集
val_portion = len(data) - train_portion - test_protion # 5%作为验证集

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_protion]
val_data = data[train_portion + test_protion:]

# print("Training set length:", len(train_data))
# print("Validation set length:", len(val_data))
# print("Test set length:", len(test_data))
# Training set length: 935
# Validation set length: 55
# Test set length: 110

##############################
# 7.3 将数据组织成训练批次
##############################

# 实现一个指令数据集类
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry) # 预词元化文本
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]

# 自定义函数实现填充
def custom_collate_draft_1(batch, pad_token_id=50256, device='cpu'):
    batch_max_length = max(len(item) + 1 for item in batch) # 找到批次中最长的序列
    input_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 删除之前额外填充的词元
        input_lst.append(inputs)

    inputs_tensor = torch.stack(input_lst).to(device) # 输入列表变成一个张量并转移到目标设备
    return inputs_tensor

input_1 = [0, 1, 2, 3, 4]
input_2 = [5, 6]
input_3 = [7, 8, 9]
batch = (
    input_1,
    input_2,
    input_3
)
# print(custom_collate_draft_1(batch))
# tensor([[    0,     1,     2,     3,     4],
#         [    5,     6, 50256, 50256, 50256],
#         [    7,     8,     9, 50256, 50256]])


def custom_collate_draft_2(batch, pad_token_id=50256, device='cpu'):
    batch_max_length = max(len(item) + 1 for item in batch) # 找到批次中最长的序列
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 删除之前额外填充的词元
        targets = torch.tensor(padded[1:]) # 像左移动一个位置得到目标
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device) # 输入列表变成一个张量并转移到目标设备
    targets_tensor = torch.stack(targets_lst).to(device) # 目标列表变成一个张量并转移到目标设备
    return inputs_tensor, targets_tensor

# inputs, targets = custom_collate_draft_2(batch)
# print(inputs)
# print(targets)
# tensor([[    0,     1,     2,     3,     4],
#         [    5,     6, 50256, 50256, 50256],
#         [    7,     8,     9, 50256, 50256]])

# tensor([[    1,     2,     3,     4, 50256],
#         [    6, 50256, 50256, 50256, 50256],
#         [    8,     9, 50256, 50256, 50256]])


def custom_collate_fn(
    batch, 
    pad_token_id=50256,
    ignore_index=-100, 
    allowed_max_length=None, 
    device='cpu'
):
    batch_max_length = max(len(item) + 1 for item in batch) # 找到批次中最长的序列
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # 填充词元pad_token_id
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # 删除之前额外填充的词元
        targets = torch.tensor(padded[1:]) # 像左移动一个位置得到目标

        # 把目标序列中的除第一个填充词元以外的填充词元填充为ignone_index
        mask = targets == pad_token_id # 找到所有填充 Token 的位置，例如[102, 103, 50256, 50256, 50256]，得到[False, False, True, True, True]
        indices = torch.nonzero(mask).squeeze() # 找出 True 所在的索引，即[2, 3, 4]
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index # 保留第一个填充 Token，将其余位置设为 -100

        # 可选的截断至最大序列长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device) # 输入列表变成一个张量并转移到目标设备
    targets_tensor = torch.stack(targets_lst).to(device) # 目标列表变成一个张量并转移到目标设备
    return inputs_tensor, targets_tensor

# inputs, targets = custom_collate_fn(batch)
# print(inputs)
# print(targets)
# tensor([[    0,     1,     2,     3,     4],
#         [    5,     6, 50256, 50256, 50256],
#         [    7,     8,     9, 50256, 50256]])

# tensor([[    1,     2,     3,     4, 50256],
#         [    6, 50256,  -100,  -100,  -100],
#         [    8,     9, 50256,  -100,  -100]])

# 计算损失案例
logits_1 = torch.tensor(
    [
        [-1.0, 1.0],
        [-0.5, -1.0]
    ]
)
targets_1 = torch.tensor([0, 1]) # 要生成的正确词元索引
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
# print(loss_1)
# tensor(1.5505)

logits_2 = torch.tensor(
    [
        [-1.0, 1.0],
        [-0.5, -1.0],
        [-0.5, -1.0]
    ]
)
targets_2 = torch.tensor([0, 1, 1]) # 要生成的正确词元索引
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
# print(loss_2)
# tensor(1.3584)

# 计算损失案例(-100,计算损失时忽略)
logits_3 = torch.tensor(
    [
        [-1.0, 1.0],
        [-0.5, -1.0],
        [-0.5, -1.0]
    ]
)
targets_3 = torch.tensor([0, 1, -100]) # 要生成的正确词元索引
loss_3 = torch.nn.functional.cross_entropy(logits_3, targets_3)
# print(loss_3)
# print("loss_1 == loss_3:", loss_1 == loss_3)
# tensor(1.5505)
# loss_1 == loss_3: tensor(True)

##############################
# 7.4 创建指令数据集的数据加载器
##############################

device = torch.device('cuda' if  torch.cuda.is_available()  else 'cpu')
# apple 芯片
# if torch.backends.map.is_available():
#     device = torch.device("mps")
# print('Device:', device)
# Device: cpu

from functools import partial
#为了custom_collate_fn函数应用于DataLoader类时重用所选择的device，利用partial预先填充设备参数
custom_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

from torch.utils.data import DataLoader
# 初始化数据加载器
num_workers = 0
batch_size = 8

torch.manual_seed(123)
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    collate_fn = custom_collate_fn,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    collate_fn = custom_collate_fn,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = custom_collate_fn,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)

# print("Train loader:")
# print(len(train_loader))
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)

# Train loader:
# 116
# torch.Size([8, 61]) torch.Size([8, 61])
# torch.Size([8, 76]) torch.Size([8, 76])
# torch.Size([8, 73]) torch.Size([8, 73])
# ...

##############################
# 7.5 加载预训练的模型GPT2-355M
##############################
# 跨目录import
import sys
from pathlib import Path
# 1.获取当前文件所在目录的父目录
project_root = Path(__file__).resolve().parent.parent
# 2.将根目录加入 Python 模块搜索路径
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from model.gpt_model import GPTModel
from configs.config import GPT_CONFIG_355M
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_inference import generate, text_to_token_ids, token_ids_to_text

model = GPTModel(GPT_CONFIG_355M)
load_gpt2_weights_into_model(model, '../weights/pytorch_model_355m.bin')
model.eval()

torch.manual_seed(123)
input_text = format_input(val_data[0])
# print(input_text)
# Below is an instruction that describes a task.Write a response that appropriately completes the request.

# ### Instruction:
# Convert the active sentence to passive: 'The chef cooks the meal every day.'

# 测试未经指令微调的模型 指令遵从能力
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=GPT_CONFIG_355M['context_length'],
    eos_id=50256,
)
generate_text = token_ids_to_text(token_ids, tokenizer)

# 减去输入指令的长度
response_text = generate_text[len(input_text):].strip()
# print(response_text)
# ### Instruction:

# Convert the active sentence to passive: 'The chef cooks the meal every day.'

# ### Instruction:

# Convert the active


##############################
# 7.6 在指令数据上微调模型
##############################
from utils.metrics import calc_loss_loader
from utils.train_model import train_model

# 微调前计算训练集和验证集的初始损失
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)
# Training loss: 4.0135581493377686
# Validation loss: 3.9755155563354494


# 对基础模型进行指令微调
import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 2
train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Ep 1 (Step 000000): Train loss 2.776, Val loss 2.750
# Ep 1 (Step 000005): Train loss 1.117, Val loss 1.144
# Ep 1 (Step 000010): Train loss 0.931, Val loss 0.951
# Ep 1 (Step 000015): Train loss 0.825, Val loss 0.927
# Ep 1 (Step 000020): Train loss 0.803, Val loss 0.853
# Ep 1 (Step 000025): Train loss 0.770, Val loss 0.909
# Ep 1 (Step 000030): Train loss 0.713, Val loss 0.810
# Ep 1 (Step 000035): Train loss 0.753, Val loss 0.845
# Ep 1 (Step 000040): Train loss 0.666, Val loss 0.797
# Ep 1 (Step 000045): Train loss 0.771, Val loss 0.731
# Ep 1 (Step 000050): Train loss 0.696, Val loss 0.744
# Ep 1 (Step 000055): Train loss 0.608, Val loss 0.753
# Ep 1 (Step 000060): Train loss 0.657, Val loss 0.750
# Ep 1 (Step 000065): Train loss 0.661, Val loss 0.723
# Ep 1 (Step 000070): Train loss 0.540, Val loss 0.712
# Ep 1 (Step 000075): Train loss 0.562, Val loss 0.737
# Ep 1 (Step 000080): Train loss 0.670, Val loss 0.738
# Ep 1 (Step 000085): Train loss 0.575, Val loss 0.722
# Ep 1 (Step 000090): Train loss 0.532, Val loss 0.662
# Ep 1 (Step 000095): Train loss 0.623, Val loss 0.727
# Ep 1 (Step 000100): Train loss 0.548, Val loss 0.660
# Ep 1 (Step 000105): Train loss 0.430, Val loss 0.607
# Ep 1 (Step 000110): Train loss 0.523, Val loss 0.687
# Ep 1 (Step 000115): Train loss 0.450, Val loss 0.631
# Below is an instruction that describes a task.Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The meal is prepared every day by the chef.<|endoftext|>The following is an instruction that describes a task.Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive:
# Ep 2 (Step 000120): Train loss 0.500, Val loss 0.642
# Ep 2 (Step 000125): Train loss 0.460, Val loss 0.663
# Ep 2 (Step 000130): Train loss 0.453, Val loss 0.663
# Ep 2 (Step 000135): Train loss 0.459, Val loss 0.617
# Ep 2 (Step 000140): Train loss 0.457, Val loss 0.716
# Ep 2 (Step 000145): Train loss 0.456, Val loss 0.729
# Ep 2 (Step 000150): Train loss 0.375, Val loss 0.658
# Ep 2 (Step 000155): Train loss 0.438, Val loss 0.647
# Ep 2 (Step 000160): Train loss 0.448, Val loss 0.663
# Ep 2 (Step 000165): Train loss 0.350, Val loss 0.671
# Ep 2 (Step 000170): Train loss 0.417, Val loss 0.678
# Ep 2 (Step 000175): Train loss 0.395, Val loss 0.651
# Ep 2 (Step 000180): Train loss 0.389, Val loss 0.695
# Ep 2 (Step 000185): Train loss 0.384, Val loss 0.669
# Ep 2 (Step 000190): Train loss 0.384, Val loss 0.648
# Ep 2 (Step 000195): Train loss 0.403, Val loss 0.636
# Ep 2 (Step 000200): Train loss 0.352, Val loss 0.591
# Ep 2 (Step 000205): Train loss 0.370, Val loss 0.644
# Ep 2 (Step 000210): Train loss 0.351, Val loss 0.623
# Ep 2 (Step 000215): Train loss 0.307, Val loss 0.642
# Ep 2 (Step 000220): Train loss 0.321, Val loss 0.622
# Ep 2 (Step 000225): Train loss 0.308, Val loss 0.600
# Ep 2 (Step 000230): Train loss 0.350, Val loss 0.648
# Below is an instruction that describes a task.Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The chef cooks the meal every day.<|endoftext|>The following is an instruction that describes a task.Write a response that appropriately completes the request.  ### Instruction: Rewrite the sentence using a simile. 
# Training completed in 3.35 minutes.

#绘制图表
from previous_chapters import plot_losses
# Alternatively:
# from llms_from_scratch.ch05 import plot_losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# # 保存模型权重
# model_path = "i../weights/nstruction_executor.pth"
# torch.save(model.state_dict(), model_path)

##############################
# 7.7 抽取并保存模型回复
##############################
# 测试三个样本的预期回复和模型回复对比

#加载权重
model_path="../weights/instruction_executor.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))

torch.manual_seed(123)
for entry in test_data[:3]:
  input_text = format_input(entry)
  token_ids = generate(
      model=model,
      idx=text_to_token_ids(input_text, tokenizer).to(device),
      max_new_tokens=256,
      context_size=GPT_CONFIG_355M['context_length'],
      eos_id=50256
  )
  generated_text = token_ids_to_text(token_ids, tokenizer)
  response_text = (
      generate_text[len(input_text):].replace("### Response:", "").strip()
  )
  print(input_text)
  print(f"\nCorrect response:\n>> {entry['output']}")
  print(f"\nModel response:\n>> {response_text.strip()}")
  print("------------------------------------------------")
# Below is an instruction that describes a task.Write a response that appropriately completes the request.

# ### Instruction:
# Rewrite the sentence using a simile.

# ### Input:
# The car is very fast.

# Correct response:
# >> The car is as fast as lightning.

# Model response:
# >> The car is as fast as a cheetah.
# ---------------------------------------------------------------------
# Below is an instruction that describes a task.Write a response that appropriately completes the request.

# ### Instruction:
# What type of cloud is typically associated with thunderstorms?

# Correct response:
# >> The type of cloud typically associated with thunderstorms is cumulonimbus.

# Model response:
# >> The type of cloud typically associated with thunderstorms is a cumulus cloud.
# ---------------------------------------------------------------------
# Below is an instruction that describes a task.Write a response that appropriately completes the request.

# ### Instruction:
# Name the author of 'Pride and Prejudice'.

# Correct response:
# >> Jane Austen.

# Model response:
# >> The author of 'Pride and Prejudice' is Jane Austen.
# ---------------------------------------------------------------------

# 生成测试集上的回复
from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=GPT_CONFIG_355M['context_length'],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text


with open("instruction-test-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    
# 100%|██████████| 110/110 [01:44<00:00,  1.05it/s]

# print(test_data[0])
# {'instruction': 'Rewrite the sentence using a simile.', 'input': 'The car is very fast.', 'output': 'The car is as fast as lightning.', 'model_response': 'The car is as fast as a cheetah.'}

##############################
# 7.8 评估指令微调后的模型
##############################
# 主要通过短答案、多项选择、MMLU基准测试考察模型基础知识，通过chatbot竞技场进行人类偏好比较

# 下载Ollama的Llam3—70B模型或通过GPT4 API
# 组装测试集以及回复请求模型评价以及打分
# 根据分数平均值动态调整批次大小、轮次、学习率等参数
