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

