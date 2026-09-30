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

    inpute_text = (
        f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    )
    return instruciton_text + inpute_text

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
