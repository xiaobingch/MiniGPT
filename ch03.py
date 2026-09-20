#3.3 无可训练权重的简单注意力机制
import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1] #jonrney 作为查询向量
# 计算注意力分数
attn_scores_2 = torch.empty(inputs.shape[0]) #创建一个与词元向量元素个数相同的初始化张量
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)#查询向量与其他元素通过点积运算求出主意力分数
# print(attn_scores_2)

# 归一化注意力分数,计算出注意力权重
attn_weight_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weight_2_tmp)
# print("Sum:", attn_weight_2_tmp.sum())
# Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
# Sum: tensor(1.0000)

# 自实现softmax归一化函数 ,优化剃度，保证权重是正值
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("注意力权重: ", attn_weights_2_naive)
# print("权重之和:", attn_weights_2_naive.sum())

# 注意力权重的Pytorch实现，增加稳定性
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attentions weights:",attn_weights_2)
# print("Sum:", attn_weights_2.sum())

#计算上下文向量
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
# print(context_vec_2)

#计算所有token的注意力分数
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)

#通过矩阵乘法计算注意力分数
attn_scores = inputs @ inputs.T
# print(attn_scores)
#对每一行进行归一化
attn_weights = torch.softmax(attn_scores, dim=-1)#dim=-1 表示沿 Tensor 的最后一个维度进行 softmax。对于二维矩阵，就是沿每一行的列元素计算，因此每一行的元素之和为 1
# print(attn_weights)

#通过矩阵乘法用这些注意力权重计算所有上下文向量
all_context_vece = attn_weights @ inputs
# print(all_context_vece)


# 3.4 实现带可训练权重的自注意力机制（缩放点积注意力）
x_2 = inputs[1]
d_in = inputs.shape[1]#嵌入维度3
d_out = 2#输出嵌入维度 2
#a.初始化三个权重矩阵q、k、v
torch.manual_seed(123)
#requires_grad训练时需要设置为true以便更新权重矩阵
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# b.通过矩阵乘法计算所有健向量和值向量
# 单个计算
query_2 = x_2 @ W_query
key_2  = x_2 @ W_key
value_2 = x_2 @ W_value
# print(query_2)
# 批量计算
keys = inputs @ W_key
values = inputs @ W_value
# print('inputs:', inputs)
# print('keys shape:', keys)
# print('valuse shape:', values)


# c. 通过矩阵乘法计算所有注意力分数，通过计算query向量和kye向量的点积来获取
# 单个计算
key_2 = keys[1]
attn_scores_22 = query_2.dot(key_2)
# print(attn_scores_22)
# 批量计算
attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

# d. 缩放注意分力分数并应用softmax函数计算注意力权重
# 公式：（注意力分数/键向量嵌入维度）的平方根【等同以0.5为指数进行幂运算】，所以也称缩放点积注意力机制
d_k = keys.shape[-1]
# print(d_k)
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)
# print(attn_weights_2.sum())

# e. 最后是计算上下文向量，value向量根据注意力权重加权计算
context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)

# 3.4.2 一个简化的自注意力python类
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

# 使用pytorch线性层的自注意力类,优化了偏置单元被禁用时权重初始化方案，提升稳定性和有效性
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))# 因为采用了更复杂的初始化方案，所以输出一致

# 3.5 利用因果注意力（掩码注意力）隐藏未来词汇
# 因果注意力掩码实现
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5, dim=-1
)
# print(attn_weights)
context_length = attn_scores.shape[0]
# print(context_length)
#a.创建一个对角线以上元素为0的掩码
mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)
# tensor([[1., 0., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1.]])
# b. 权重矩阵与掩码相乘
mask_simple = attn_weights * mask_simple
# print(mask_simple)
# c. 重新归一化权重
row_sums = mask_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = mask_simple / row_sums
# print(masked_simple_norm)

# d. 在注意力分数阶段，归一化之前先掩码再softMax归一化可以简化一个归一化步骤
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1) #掩码1
masked = attn_scores.masked_fill(mask.bool(), -torch.inf) #将1替换为负无穷-inf
# print(masked)
attn_weights = torch.softmax(
    masked / keys.shape[-1]**0.5, dim=1
)
# print(attn_weights)
# context_vec = attn_weights @ values

# 3.5.2 掩码额外的注意力权重
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)#使用50%的dropout率
example = torch.ones(6, 6)#创建一个全1矩阵
# print(dropout(example))#50%的值被置0，其他元素会按1/0.5的比例进行放大
torch.manual_seed(123)
# print(dropout(attn_weights))
# 3.5.3 实现一个简化的因果注意力类
batch = torch.stack((inputs, inputs), dim=0)#2个批次、6个词元/批次、嵌入维度3
# print(batch.shape)
# torch.Size([2, 6, 3])
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)#dropout掩码层
        self.register_buffer(#使用这个类时，缓冲区会与模型一起自动移动到适当的设备（cpu或gpu），避免设备不匹配的错误
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)#将维度1和2转置，将批次维度保持在第一位
        #pytorch中所有_结尾的操作直接作用于原数据，减少不必要的内存复制
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
        

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print(context_vecs)
# print('context_vecs.shape:', context_vecs.shape)
# context_vecs.shape: torch.Size([2, 6, 2])


# 3.6 将单头注意力扩张到多头注意力
# 一个实现多头注意力的封装类
class MutiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1]#词元的数量
d_in, d_out = 3, 2
mha = MutiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)
# context_vecs.shape: torch.Size([2, 6, 4])# 一维是2，第二维是6个词元，第三维是（2头 * 2维）四维嵌入


#一个高效的多头注意力类
class MutiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #减少投影维度以匹配所需的输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out) #使用一个线性层来组合头的输出
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        #张量形状：b, num_tokens, d_out
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        #通过num_heads隐式分割矩阵，然后展开最后一个维度，
        #新形状：b, num_tokens, num_heads, head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        #调换1、2维度的位置,新形状：b, num_heads, num_tokens, head_dim
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        #计算每个头的点积
        attn_scores = queries @ keys.transpose(2,3)
        #被截断为词元数量的掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        #使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        #张量形状：b, num_tokens, num_heads, head_dim
        context_vec = (attn_weights @ values).transpose(1,2)
        #组合头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)#添加一个可选线性投影
        return context_vec

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MutiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
# print(context_vecs)
# print('context_vecs.shape:', context_vecs.shape)

mha = MutiHeadAttention(768, 768, 3, 0.0, 12)
print(mha)