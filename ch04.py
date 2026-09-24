# 4.1 构建一个大语言模型
GPT_CONFIG_124M = {
    'vocab_size': 50257,        #词汇表大小
    # 'context_length': 1024,     #上下文长度
    'context_length': 256,     #上下文长度
    'emb_dim':768,              #嵌入维度
    'n_heads':12,               #注意力头数
    'n_layers':12,              #层数
    'drop_rate':0.1,            #dropout率
    'qkv_bias':768,             #查询-键-值偏置
}

# GPT模型架构
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])]#TransformerBlock 占位
        )
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):#占位类
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):#占位类
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

# 分词
import tiktoken


tokenizer = tiktoken.get_encoding('gpt2')
batch= []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)
# Output shape: torch.Size([2, 4, 50257])
# tensor([[[-0.9289,  0.2748, -0.7557,  ..., -1.6070,  0.2702, -0.5888],
#          [-0.4476,  0.1726,  0.5354,  ..., -0.3932,  1.5285,  0.8557],
#          [ 0.5680,  1.6053, -0.2155,  ...,  1.1624,  0.1380,  0.7425],
#          [ 0.0447,  2.4787, -0.8843,  ...,  1.3219, -0.0864, -0.5856]],

#         [[-1.5474, -0.0542, -1.0571,  ..., -1.8061, -0.4494, -0.6747],
#          [-0.8422,  0.8243, -0.1098,  ..., -0.1434,  0.2079,  1.2046],
#          [ 0.1355,  1.1858, -0.1453,  ...,  0.0869, -0.1590,  0.1552],
#          [ 0.1666, -0.8138,  0.2307,  ...,  2.5035, -0.3055, -0.3083]]],
#        grad_fn=<UnsafeViewBackward0>)

# 4.2 使用层归一化进行归一化激活，ReLU: 均值为0方差为1，加速权重的有效收敛提高训练过程的一致性和可靠性
# 方差：(每个数据与均值的差的平方之和) ÷ 数据总个数
# 均值：所有数据之和 ÷ 数据总个数
torch.manual_seed(123)
batch_example = torch.randn(2, 5)#2个训练样本5个维度（特征）
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())#非线性激活函数ReLU
out = layer(batch_example)
# print(out)
# tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
#        grad_fn=<ReluBackward0>)
# 检查均值和方差
mean = out.mean(dim=-1, keepdim = True)
var = out.var(dim=-1, keepdim = True)
# print("Mean:\n", mean)
# print("Variance:\n", var)
# Mean:
#  tensor([[0.1324],
#         [0.2170]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[0.0231],
#         [0.0398]], grad_fn=<VarBackward0>)
# 层输出归一化操作:减去均值/方差的平方根
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
# print("Normalized layer outputs:\n", out_norm)
# print("Mean:\n", mean)
# print("Variance:\n", var)
# Normalized layer outputs:
#  tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
#         [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
#        grad_fn=<DivBackward0>)
# Mean:无限接近0
#  tensor([[9.9341e-09],
#         [0.0000e+00]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)
# torch.set_printoptions(sci_mode=False)#关闭科学记数法
# print("Mean:\n", mean)
# print("Variance:\n", var)
# Mean:
#  tensor([[    0.0000],
#         [    0.0000]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)

# print("out:\n", out)
# print("out_norm:\n",out_norm)

# 层归一化类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)
# print("Out layer norm:\n", out_ln)
# Mean:
#  tensor([[    -0.0000],
#         [     0.0000]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)
# Out layer norm:
#  tensor([[ 0.5528,  1.0693, -0.0223,  0.2656, -1.8654],
#         [ 0.9087, -1.3767, -0.9564,  1.1304,  0.2940]], grad_fn=<AddBackward0>)

# 4.3 实现具有GELU（高斯误差线性单元）激活函数的前馈神经网络，比ReLU（修正线性单元）更平滑且复杂
class  GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )
# 对比GELU 和 ReLU激活函数
# import matplotlib.pyplot as plt
# gelu, relu = GELU(), nn.ReLU()
# x = torch.linspace(-3, 3, 100)#在-3 和3 之间创建100个样本数据点
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f'{label} activation function')
#     plt.xlabel('x')
#     plt.ylabel(f'{label}(x)')
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

# 4.4 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
# print(out.shape)
# torch.Size([2, 3, 768])

# 4.4 添加快捷连接（跳跃连接或残差连接）
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# 有无快捷连接的梯度对比
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1,]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

def print_gradients(model, x):
    output = model(x)#前向传播
    target = torch.tensor([[0.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)#计算损失

    loss.backward()#反向传播

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# print_gradients(model_without_shortcut, sample_input)#梯度消失，反向传播过程中梯度逐渐缩小
# layers.0.0.weight has gradient mean of 0.00020173587836325169
# layers.1.0.weight has gradient mean of 0.0001201116101583466
# layers.2.0.weight has gradient mean of 0.0007152041653171182
# layers.3.0.weight has gradient mean of 0.001398873864673078
# layers.4.0.weight has gradient mean of 0.005049646366387606
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
# print_gradients(model_without_shortcut, sample_input)#梯度恢复正常
# layers.0.0.weight has gradient mean of 0.22169792652130127
# layers.1.0.weight has gradient mean of 0.20694106817245483
# layers.2.0.weight has gradient mean of 0.32896995544433594
# layers.3.0.weight has gradient mean of 0.2665732502937317
# layers.4.0.weight has gradient mean of 1.3258541822433472
    
# 4.5连接Traunsformer块的注意力层和线性层
# GPT的Transformer块组件
from ch03 import MutiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MutiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x #注意力模块添加快捷连接
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut #原始输入添加回来

        shortcut = x #前馈层添加快捷连接
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # 原始输入添加回来
        return x

torch.manual_seed(123)
x = torch.rand(2, 4, 768) #[batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

# print("Iputs shape:\n", x.shape)
# print("Outputs shape:\n", output.shape)
# Iputs shape:
#  torch.Size([2, 4, 768])
# Outputs shape:
#  torch.Size([2, 4, 768])


# 4.7 实现GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)#device的实质允许我们选择cpg或gpt序列模型
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch) #[batch_size, num_tokens, vocab_size]
# print('input batch:\n', batch)
# print('output shape:\n',out.shape)
# print(out)
# input batch:
#  tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])
# output shape:
#  torch.Size([2, 4, 50257])
# tensor([[[-0.3348, -0.3230,  0.0295,  ...,  0.2345, -1.2997,  0.2420],
#          [ 0.4105, -0.8302,  1.7490,  ..., -1.0524, -2.0505,  0.8429],
#          [ 0.9421, -0.5276, -0.5335,  ..., -0.2873, -1.0361, -0.7200],
#          [ 0.2697,  0.8770,  0.5189,  ...,  0.5305,  0.4347,  0.5527]],

#         [[-1.5011, -0.9993, -0.1497,  ..., -0.9591, -1.0575,  0.0236],
#          [ 0.4101, -2.9044,  2.4342,  ..., -1.8442, -1.4876,  0.4186],
#          [-1.0829, -0.1956, -0.2907,  ..., -0.5564, -0.6518,  0.7109],
#          [-0.2047,  2.0846,  1.1762,  ..., -0.9225, -0.5120, -0.8725]]],
#        grad_fn=<UnsafeViewBackward0>)

#计算参数量，词元嵌入与输出头的权重共享只计算一次
total_params = sum(p.numel() for p in model.parameters())
# print(f'Total number of parameters: {total_params:,}')
# Total number of parameters: 163,035,648
# 形状相同
# print("Token embedding layer shape:",model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)
# Token embedding layer shape: torch.Size([50257, 768])
# Output layer shape: torch.Size([50257, 768])
# 计算最终参数量
total_pramas_gpt2 = (
    total_params - sum(p.numel() for p in model.out_head.parameters())
)
# print(f'Number of trainable parameters considring weight tying:{total_pramas_gpt2:,}')
# Number of trainable parameters considring weight tying:124,438,272
# 注意力模块与前馈层参数量计算
total_trf_params = sum(
    p.numel() for p in model.trf_blocks.parameters()
)
# print(f"Number of TransformerBlock parameters:{total_trf_params:,}")
# Number of TransformerBlock parameters:85,054,464

#计算参数所需内存
total_size_bytes = total_params * 4 #假设每个参数占用4个字节的32位浮点数
tatal_size_mb = total_size_bytes / (1024 * 1024)
# print(f"Total size of model:{tatal_size_mb:.2f} MB")
# Total size of model:621.93 MB

# 4.7 生成文本
 #idx：当前文本的索引数组[batch, n_tokens]
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] #截取符合最大长度的token
        with torch.no_grad():
            logits = model(idx_cond)
        # 只关注最后一个输出的token
        # 形状变换：[batch, n_tokens, vocab_size]---> [batch, vocab_size]
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        # 形状变为：[batch, 1]
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # 将新token的索引添加到索引数组重，形状变为：[batch, n_tokens + 1]
        idx = torch.cat((idx, idx_next), dim=1) 

    return idx
        
start_context = 'Hello, I am'
encode = tokenizer.encode(start_context)
encode_tensor = torch.tensor(encode).unsqueeze(0) # 添加batch维度
# print('encoded:', encode)
# print("encoded_tensor.shape:", encode_tensor.shape)
# encoded: [15496, 11, 314, 716]
# encoded_tensor.shape: torch.Size([1, 4])

model.eval()#不训练时关闭dropout
out = generate_text_simple(
    model=model,
    idx=encode_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())

# print('Output:', out)
# print('Output length:', len(out[0]))
# print(decoded_text)
# Output: tensor([[15496,    11,   314,   716,  3127, 29991,  6539, 21826, 18530,  6276]])
# Output length: 10
# Hello, I am network BEL Afghan postp aired technical