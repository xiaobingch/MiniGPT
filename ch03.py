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
#初始化三个权重矩阵q、k、v
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)#requires_grad训练时需要设置为true以便更新权重矩阵
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)



