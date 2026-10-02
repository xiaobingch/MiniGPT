import tiktoken
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    '''
    用于批处理输入和目标的数据集
    Args:
        txt:文本数据
        tokenizer:分词器
        max_length:序列最大长度
        stride:滑动窗口的步长
    '''
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口方法将文本划分为长度为max_length的重叠序列
        # stride为步长，决定了每个序列之间的重叠部分
        # 例如，max_length=10, stride=5时，序列为[0:10], [5:15], [10:20]等
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 返回数据集的总行数
    def __len__(self):
        return len(self.input_ids)

    # 返回数据集的指定行
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, stride,shuffle=True, drop_last=True, num_workers=0):
    '''
    用于批量生成输入-目标对的数据加载器
    Args:
        txt: 文本数据
        batch_size: 批次大小
        max_length: 序列最大长度
        stride: 滑动窗口的步长
        shuffle: 是否打乱数据顺序 防止模型过拟合
        drop_last: 是否丢弃最后一个不完整的batch
        num_workers: 多线程加载数据的工作线程数
    '''
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class SpamDataset(Dataset):
    '''
    批处理分类微调数据集的输入(文本内容)和目标(类别标签)
    Args:
        data: pandas DataFrame 格式的数据集
        tokenizer: 分词器
        max_length: 每个序列的最大长度
        pad_token_id: 填充 token 的 id, 默认使用 GPT-2 的 token id 50256 即 <|endoftext|> 来填充
    '''
    def __init__(self, data, tokenizer, max_length=None, pad_token_id=50256):
        self.data = data

        # 将文本数据编码为 token id
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            # 获取数据集中最长的序列长度
            self.max_length = self._longest_encoded_length()
            #等同于_longest_encoded_length方法
            # self.max_length = max(len(encoded_text) for encoded_text in self.encoded_texts)
        else:
            self.max_length = max_length
            # 如果编码后的文本长度超过 max_length，则截断
            # 比如模型的最大上下文长度是 1024，那么如果文本长度超过 1024，则截断到 1024
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        # 如果编码后的文本长度小于 max_length，则填充到 max_length
        # 使用 gpt2 tokenizer 的 token id 50256 即 <|endoftext|> 来填充
        # 确保每个输入张量的大小相同，以便通过 PyTorch DataLoader 加载数据集
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    # 返回数据集的指定行的数据
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"] # 0非垃圾消息，1垃圾消息
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    # 返回数据集的总长度
    def __len__(self):
        return len(self.data)
    
    # 查找数据集中序列的最大长度
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


class InstructionDataset(Dataset):
    '''
    用于加载指令微调数据集的类

    Args:
        data: 字符串格式的文本数据
        tokenizer: 分词器
    '''
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

def custom_collate_fn(
    batch, 
    pad_token_id=50256,
    ignore_index=-100, 
    allowed_max_length=None, 
    device='cpu'
):
    '''
    指令微调的自定义批处理函数，用于自定义批次数据的整理方式
    有自定义 collate_fn 的情况下，随机/不随机（shuffle）选择 batch 个索引传入 dataset 里的 __getitem__(self, index) 得到对应的数据，
    将这些数据（样本对）传入 collate_fn 指定函数进行处理
    
    Args:
        batch: 当前批次的数据
        pad_token_id: 填充 token 的 id, 默认使用 GPT-2 的 token id 50256 即 <|endoftext|> 来填充
        ignore_index: 忽略的 id, 在计算损失值时将不计算包含这些 id 的 token
        allowed_max_length: 允许的最大序列长度，如超过则截断
        device: 决定模型在 CPU 还是 GPU 上运行
    '''
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