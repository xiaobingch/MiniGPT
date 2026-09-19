import urllib.request
# url = ("https://raw.githubuserconten.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)
# 原始数据集
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# 2-1文本分词
import re
text = "hello, word. This, is a tset."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30])

# 2-2创建词汇表
all_words = sorted(set(preprocessed))
vacab_size = len(all_words)
# print(vacab_size)

vacab = {token:integer for integer,token in enumerate(all_words)}
for i,item in enumerate(vacab.items()):
    # print(item)
    if i > 50:
        break
# print(vacab)

# 2-3简单文本分词器
class SimpleTokenizerV1:
    def __init__(self, vacab):
        self.str_to_int = vacab #词汇表token:id
        self.int_to_str = {i:s for s,i in vacab.items()}#逆向词汇表id:token
    
     # 将文本转换为词元ID
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

     # 将id转换为文本
    # def decode(self, ids):
    #     text = " ".join([self.int_to_str[i] for i in ids])
    #     text = re.sub(r'\s+[,.?!"()\']', r'\1', text) #移除标点符号前的空格
    #     return text

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.;?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vacab)
text = """even through the prism"""
ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))
text = "hello, do you like tea?"
# print(tokenizer.encode(text))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vacab  = {token:integer for integer,token in enumerate(all_tokens)}
# print(len(vacab.items()))


for i, item in enumerate(list(vacab.items())[-5:]):
    # print(item)
    pass


# 2-4能够处理未知单词文本分词器
class SimpleTokenizerV2:
    def __init__(self, vacab):
        self.str_to_int = vacab #词汇表token:id
        self.int_to_str = {i:s for s,i in vacab.items()}#逆向词汇表id:token
    
     # 将文本转换为词元ID
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

     # 将id转换为文本
    # def decode(self, ids):
    #     text = " ".join([self.int_to_str[i] for i in ids])
    #     text = re.sub(r'\s+[,.?!"()\']', r'\1', text) #移除标点符号前的空格
    #     return text

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))
# print(text)


tokenizer = SimpleTokenizerV2(vacab)
ids = tokenizer.encode(text)

# print(tokenizer.decode(ids))


# 2.5 BPE分词器
from importlib.metadata import version
import tiktoken
# print(version('tiktoken'))
tokenizer = tiktoken.get_encoding('gpt2')
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
strings = tokenizer.decode(integers)
# print(strings)


# 2.6 使用滑动窗口进行数据采样
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
# print(f'x: {x}')
# print(f'y:      {y}')

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(context, '--->', desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(tokenizer.decode(context), '--->', tokenizer.decode([desired]))


# import torch
# torch.__version__
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
# 一个用于批处理输入和目标的数据集
# class GPTDatasetV1(Dataset):
#     def __init__(self, txt, tokenizer, max_length, stride):
#         self.input_ids = []
#         self.target_ids = []

#         token_ids = tokenizer.encode(txt)#对全部文本进行分词
#         # 使用滑动窗口划分长度max_legnth、重叠采样步进为stride的序列
#         for i in range(0, len(token_ids) - max_length, stride):
#             input_chunk = token_ids[i:i + max_length]
#             target_chunk = token_ids[i+1:i+max_length+1]
#             self.input_ids.append(torch.tensor(input_chunk))
#             self.target_ids.append(torch.tensor(target_chunk))
#         # 返回数据集的总行数
#         def __len__(self):
#             return len(self.input_ids)
        
#         # 返回数据集的指定行
#         def __getitem__(slef, idx):
#             return self.input_ids[idx], self.target_ids[idx]

# # 用于批量生成输入-目标对的数据加载器
# def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
#     tokenizer = tiktoken.get_encoding('gpt2') #初始化分词器
#     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)# 创建数据集
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,#如果drop_last为true且批次大小小于指定的batch_size,则会删除追后一批，防止在训练期间出现损失剧增
#         num_workers=num_workers#处理cpu进程数
#     )
#     return dataloader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
# print(first_batch)
# second_batch = next(data_iter)
# print(second_batch)


data_iter = iter(dataloader)
inputs,targets = next(data_iter)
# print('Inputs:\n', inputs)
# print('Targets:\n', targets)



# 2.7 创建词元嵌入
input_ids = [2,3,5,1]
vacab_size = 6  #模拟词表大小
output_dim = 3 #嵌入维度

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vacab_size, output_dim)
# print(embedding_layer.weight)#底层权重矩阵
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True)
# print(embedding_layer(torch.tensor([3])))#输入词元id获取嵌入向量
# tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)

# print(embedding_layer(torch.tensor([2,3,5,1])))#根据tokenId从嵌入层的权重矩阵查找对应的向量
# Parameter containing:
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True)
# tensor([[ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-2.8400, -0.7849, -1.4096],
#         [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)




# 2.8 编码token位置信息，自注意力机制无法感知词元的位置和顺序，引入可训练的位置向量
vacab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vacab_size, output_dim)
max_length = 4
# 8批次，4词元/批次，维度256
dataloader = create_dataloader_v1(
    raw_text, batch_size = 8, max_length=max_length,
    stride=max_length, shuffle = True
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Token IDs:\n", inputs)
# print("Inputes shape:\n", inputs.shape)
# Token IDs:
#  tensor([[24818,   417,    12, 12239],
#         [  314,  3114,   379,   262],
#         [ 2156,   286,  4116,    13],
#         [  866,   262,  2119,    11],
#         [ 3363,    11,   340,   373],
#         [  198,     1,    40,  2900],
#         [  465, 14475,    13,   198],
#         [ 3081,   286,  2045,  1190]])
# Inputes shape:
#  torch.Size([8, 4])

token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)
# torch.Size([8, 4, 256]) #每个token被嵌入一个256维度的向量中

#获取位置嵌入层，contex_len * 256
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))#从0开始递增
# print(pos_embeddings.shape)
# torch.Size([4, 256])
#计算输入张量，pytorch 会在每个批次的4*256维的词元嵌入张量上添加一个4*256维度的pos_embeddings张量
input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)
# torch.Size([8, 4, 256])
