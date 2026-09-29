import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from utils.dataset_loader import SpamDataset
from model.gpt_model import GPTModel
from configs.config import GPT_CONFIG_124M
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_inference import text_to_token_ids, token_ids_to_text
from utils.train_model import train_classifier

'''
基于GPT2-124M权重的模型进行分类微调
主要参数:
    data_file_path:原始分类数据集文件路径
    config: 模型参数配置
    gpt2_model_path: 加载GPT2-124M基础模型权重路径
    model_path:微调后的模型权重保存路径
'''

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
# 准备数据集
##############################

#加载原始数据集
data_file_path = 'data/SMSSpamCollection.tsv'
df = pd.read_csv(data_file_path, sep = "\t", header=None, names = ['Label', "Text"])

#平衡数据集
num_spam = df[df["Label"] == "spam"].shape[0] # 统计 垃圾消息 样本数量
ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state = 123) # 随机采样 非垃圾消息， 使其数量与垃圾消息一致
df = pd.concat([ham_subset, df[df["Label"] == "spam"]]) #组合 垃圾消息与非垃圾消息 组成平衡数据集
df["Label"] = df["Label"].map({'ham':0, "spam":1}) # 将字符串转int
df = df.sample(frac=1, random_state=123).reset_index(drop=True) # 打乱整个Dtaframe

# 切分训练集和验证集，这里将 90% 的数据作为训练集，10% 的数据作为验证集
train_ratio = 0.9
split_idx = int(len(df) * train_ratio)
train_data = df[:split_idx]
validate_data = df[split_idx:]

# 准备及加载训练数据集和验证数据集
batch_size = 8 # 设置批次大小
num_workers = 0 # 设置数据加载器的多线程工作线程数

train_dataset = SpamDataset(
    data = train_data,
    tokenizer=tokenizer,
    max_length=None
)
val_dataset = SpamDataset(
    data = validate_data,
    tokenizer=tokenizer,
    max_length = train_dataset.max_length
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False
)


##############################
# 初始化大模型并加载GPT2-124M权重
##############################
model = GPTModel(GPT_CONFIG_124M)
gpt2_model_path = "weights/pytorch_model.bin"
load_gpt2_weights_into_model(model, gpt2_model_path)
model.eval()


##############################
# 修改模型以适用分类微调
##############################

# 由于模型经过了预训练，因此不需要微调所有层。基于神经网络的语言模型中，较低层通常捕捉了通用的语言结构和语义，适用于广泛的任务和数据集。
# 而最后几层更侧重捕捉特定任务的特征，适用于特定任务的数据集。
# 这里我们将除最后一层输出层外的所有层设置为冻结，只进行最后一层的微调。
# 冻结模型，将所有层设为不可训练
for param in model.parameters():
    param.requires_grad = False

# 评论分类任务，因此我们只需替换最后的输出层即可，
# 该层原本是将输入映射为 50257 维的向量，即词汇表的大小，
# 现在将其输出层作用改为映射为 2 维的向量，即 0/1 两类的分类器
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features = GPT_CONFIG_124M['emb_dim'],
    out_features = num_classes
)
# 为了更好的训练效果，额外将最后一个 Transformer 块和最后层归一化设置为可训练
# 新的输出层的 requires_grad 默认为 True，意味着该层是模型中唯一在训练过程中会被更新的层
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True


##############################
# 分类微调模型并保存模型权重
##############################

# 初始化优化器，优化器是用于更新模型权重参数的算法，这里使用 AdamW 算法
optimizer = torch.optim.AdamW(
    model.parameters(), # .parameters()方法返回模型的所有可训练权重参数
    lr=5e-4, # 学习率，即模型权重参数的梯度下降步长的系数，决定具体变化快慢，即 w = w - lr * dL/dw
    weight_decay=0.1 # 权重衰减，即模型权重参数的 L2 正则化系数
)

# 执行模型分类微调
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=5,
    eval_freq=50,
    eval_iter=5
)

# 保存模型权重
model_path = "weights/review_classifier.pth"
torch.save(model.state_dict(), model_path)