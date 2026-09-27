import urllib.request
import zipfile
import os
from pathlib import Path
##############################
# 6.2 准备数据集
##############################
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spm_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

#下载解压书籍集
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping Download and extraction.")
        return
    
    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
    
    # 解压文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path) # 添加 .tsv文件扩展名
    print(f"FIle download and saved as {data_file_path}")

# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

import pandas as pd
df = pd.read_csv(
    data_file_path,
    sep = "\t", 
    header=None, 
    names = ['Label', "Text"]
)
# print(df)
#      Label                                               Text
# 0      ham  Go until jurong point, crazy.. Available only ...
# 1      ham                      Ok lar... Joking wif u oni...
# 2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
# 3      ham  U dun say so early hor... U c already then say...
# 4      ham  Nah I don't think he goes to usf, he lives aro...
# ...    ...                                                ...
# 5567  spam  This is the 2nd time we have tried 2 contact u...
# 5568   ham               Will ü b going to esplanade fr home?
# 5569   ham  Pity, * was in mood for that. So...any other s...
# 5570   ham  The guy did some bitching but I acted like i'd...
# 5571   ham                         Rofl. Its true to its name
# print(df["Label"].value_counts())
# Label
# ham     4825
# spam     747
# Name: count, dtype: int64

# 创建一个平衡的数据集
def create_balanced_dataset(df):
    # 统计 垃圾消息 样本数量
    num_spam = df[df["Label"] == "spam"].shape[0]
    # 随机采样 非垃圾消息， 使其数量雨 垃圾消息 一直
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state = 123)
    #组合 垃圾消息与非垃圾消息 组成平衡数据集
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df

balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())
# Label
# ham     747
# spam    747
# Name: count, dtype: int64

# 将字符串转int
balanced_df["Label"] = balanced_df["Label"].map({'ham':0, "spam":1})

# 划分数据集
def random_split(df, train_frac, validation_frac):
    # 打乱整个Dtaframe
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # 计算拆分索引
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 拆分dataframe
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df
# 测试集比例0.2
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# 保存训练、验证、测试数据集
train_df.to_csv("train.csv", index=None)
validation_df.to_csv('validation.csv', index=None)
test_df.to_csv('test.csv', index=None)


##############################
# 6.3 创建数据加载器
##############################
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
# 使用 <|endoftext|>的词元id 50256 填充较短序列以匹配最长序列的长度
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]

import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    '''
    批处理数据集的输入(文本内容)和目标(类别标签)
    Args:
        csv_file: 数据集
        tokenizer: 分词器
        max_length: 序列最大长度
        pad_token_id: <|endoftext|>的词元ID, 50256
    '''
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        # 文本分词
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果序列长度超过max_length则进行截断
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        # 填充到最长序列的长度
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"] # 0非垃圾消息，1垃圾消息
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

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

# 训练数据集
train_dataset = SpamDataset(
    csv_file = "train.csv",
    max_length=None,
    tokenizer=tokenizer
)
# print(train_dataset)
# 120

# 验证数据集
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length = train_dataset.max_length,
    tokenizer=tokenizer
)

# 测试数据集
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length = train_dataset.max_length,
    tokenizer=tokenizer
)


from torch.utils.data import DataLoader
# 兼容大多数计算机
num_workers = 0
batch_size = 8
torch.manual_seed(123)

# 训练集加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

# 验证集加载器
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)

# 测试数据集加载器
test_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)

for input_batch, target_batch in train_loader:
    pass
# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions:", target_batch.shape)
# Input batch dimensions: torch.Size([8, 120])
# Label batch dimensions: torch.Size([8])

# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")
# 130 training batches
# 19 validation batches
# 19 test batches

##############################
# 6.4 初始化带有预训练权重的模型
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
from configs.config import GPT_CONFIG_124M
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_inference import generate, text_to_token_ids, token_ids_to_text


model = GPTModel(GPT_CONFIG_124M)
load_gpt2_weights_into_model(model, 'pytorch_model.bin')
model.eval()

# text_1 = "Every effort moves you"
# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids(text_1,tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M['context_length']
# )
# print(token_ids_to_text(token_ids, tokenizer))
# Every effort moves you forward.
# The first step is to understand the importance of your work

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate(
    model=model,
    idx=text_to_token_ids(text_2,tokenizer),
    max_new_tokens=24,
    context_size=GPT_CONFIG_124M['context_length']
)
print(token_ids_to_text(token_ids, tokenizer))
# Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'
# The following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you

