import torch
from ch04 import GPTModel
GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

import tiktoken
from ch04 import generate_text_simple
# 5.1词元和词元ID的互相转换函数
def text_to_token_ids(text, tokenizer):
    encode = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encode_tensor = torch.tensor(encode).unsqueeze(0)#增加batch维度
    return encode_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)#移除batch维度
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"]
)

# print("output text:\n", token_ids_to_text(token_ids, tokenizer))

# 计算文本损失
# inputs1 = text_to_token_ids("every effort moves",tokenizer)
# inputs2 = text_to_token_ids("I really like",tokenizer)
# inputs = torch.cat([inputs1,inputs2], dim=0)
# targets1 = text_to_token_ids(" effort moves you",tokenizer)
# targets2 = text_to_token_ids(" really like chocolate",tokenizer)
# targets = torch.cat([targets1,targets2], dim=0)
# print(targets)
inputs = torch.tensor([[16833,  3626,  6100],       # ["every effort moves"]
        [   40,  1107,   588]])                     # ["I really like"]

targets = torch.tensor([[ 3626,  6100,   345],      #[" effort moves you"]
        [ 1107,   588, 11311]])                     #[" really like chocolate"]

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) #1.概率
# print(probas.shape)#torch.Size([2, 3, 50257])批次、词元数量、词元维度（词表大小）
token_ids  =torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDS:\n", token_ids)
# print(f"Targets batch 1:\n", f"{token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1:\n", f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# Token IDS:
#  tensor([[[16657],
#          [  339],
#          [42826]],

#         [[49906],
#          [29669],
#          [41751]]])
# Targets batch 1:
#  {' effort moves you'}
# Outputs batch 1:
#   Armed heNetflix
text_idx = 0
targets_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]#2.text1目标概率
# print("Text 1:", targets_probas_1)
text_idx = 1
targets_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]#2.text2目标概率
# print("Text 2:", targets_probas_2)
# Text 1: tensor([1.8793e-04, 3.4877e-05, 6.6219e-06])
# Text 2: tensor([3.9363e-06, 1.0531e-04, 1.5254e-06])

log_probas = torch.log(torch.cat((targets_probas_1, targets_probas_2)))# 3.对数概率
# print(log_probas)
avg_log_probas = torch.mean(log_probas) #4.平均对数概率
# print(avg_log_probas)
neg_avg_log_probas = avg_log_probas * -1 #5.负平均对数概率（交叉墒）
# print(neg_avg_log_probas)
# tensor(10.9609)

#使用Pytorch的cross_entropy函数处理1-5的步骤
# print("Logits shape:", logits.shape)
# print("Targets shape:", targets.shape)
# Logits shape: torch.Size([2, 3, 50257])
# Targets shape: torch.Size([2, 3])
# 展平张量维度
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)
# Flattened logits: torch.Size([6, 50257])
# Flattened targets: torch.Size([6])
# 交叉墒：0-无限大，越低预测越准确
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)
# tensor(10.9609)
# 困惑度:模型预测的概率分布于数据集实际词汇分布的匹配程度
perplexity = torch.exp(loss)
# print(perplexity)
# tensor(57577.9297)

# 计算训练集和验证集的损失
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)
# Characters: 20479
# Tokens: 5145

#切分训练集和验证集
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from  ch02 import create_dataloader_v1
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    # max_length=GPT_CONFIG_124M["context_length"],
    # stride=GPT_CONFIG_124M["context_length"],
    max_length=256,
    stride=256,
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    # max_length=GPT_CONFIG_124M["context_length"],
    # stride=GPT_CONFIG_124M["context_length"],
    max_length=256,
    stride=256,
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)
#计算训练集加载器和验证集加载器返回的给定批次的交叉墒损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        # 抹平第 0 维即 batch_size 维，将 logits 的形状从 (batch_size, num_tokens, vocab_size) 转换为 (batch_size * num_tokens, vocab_size)
        logits.flatten(0, 1), 

        # 抹平第 0 维即 batch_size 维，将 target_batch 的形状从 (batch_size, num_tokens) 转换为 (batch_size * num_tokens)
        target_batch.flatten()
    )
    return loss

#计算训练集和验证集损失的函数
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
# with torch.no_grad():#关闭梯度追踪
    # train_loss  = calc_loss_loader(train_loader, model, device)
    # val_loss = calc_loss_loader(val_loader, model, device)
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)
# Training loss: 11.240845574273003
# Validation loss: 11.230772018432617

# 5.2 训练大语言模型
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], [] #初始化列表以跟踪损失和所见的词元
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):#开始主训练循环
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()#重制上一个批次迭代中的损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()#计算损失梯度
            optimizer.step()#使用损失梯度更新模型权重
            tokens_seen += input_batch.numel() #已阅读token数量
            global_step += 1 #进行多少次参数更新

            if global_step % eval_freq == 0: #可选的评估步骤，每几步评估一次
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}):"
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

# 评估模型损失（数值）
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # 评估阶段禁用dropout以产出稳定且可复现的结果
    with torch.no_grad(): #评估阶段禁用梯度跟踪，减少计算开销也无必要
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# 评估模型（文本）
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encode = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encode, max_new_tokens=50, context_size=context_size)
    decode_text = token_ids_to_text(token_ids, tokenizer)
    print(decode_text.replace("\n", ""))
    model.train()

# 训练模型
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )

# 训练集和验证集损失图表
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots(figsize=(5, 3))

#     # Plot training and validation loss against epochs
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

#     # Create a second x-axis for tokens seen
#     ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
#     ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
#     ax2.set_xlabel("Tokens seen")

#     fig.tight_layout()  # Adjust layout to make room
#     plt.savefig("loss-plot.pdf")
#     plt.show()

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# 5.3 文本生成策略（解码策略）
model.to("cpu")
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
#     model= model,
#     idx = text_to_token_ids("Everry effort moves you", tokenizer),
#     max_new_tokens = 25,
#     context_size = GPT_CONFIG_124M['context_length']
# )
# print("Optput text:\n", token_ids_to_text(token_ids, tokenizer))

vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 

inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id])

#multinomial 代替argmax，根据概率分数采样下一个词元
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
# print(inverse_vocab[next_token_id])

def print_samled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# print_samled_tokens(probas)
# 71 x closer
# 2 x every
# 0 x effort
# 544 x forward
# 2 x inches
# 1 x moves
# 0 x pizza
# 376 x toward
# 4 x you

# 温度缩放，logits / 大于0的数，温度大于1概率分布更加均匀，小于1概率陡峭自信，缺点容易生成不相关的词语
def softmax_width_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]
scaled_probas = [softmax_width_temperature(next_token_logits, T) for T in temperatures]
# print(scaled_probas)

# Plotting
# x = torch.arange(len(vocab))
# bar_width = 0.15

# fig, ax = plt.subplots(figsize=(5, 3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

# ax.set_ylabel('Probability')
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()

# plt.tight_layout()
# plt.savefig("temperature-plot.pdf")
# plt.show()

# TOP-k采样，替换所有非前K个的logits为负无穷（概率分数为0），剩余的的概率综合为1
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# print("Top logits:", top_logits)
# print("Top positions:", top_pos)
# Top logits: tensor([6.7500, 6.2800, 4.5100])
# Top positions: tensor([3, 7, 0])
new_logits = torch.where(
    condition = next_token_logits < top_logits[-1], #识别出比前三个logits最低值还低的logits值
    input = torch.tensor(float('-inf')),#给这些更低的值填充-inf
    other = next_token_logits
)
# print(new_logits)
# tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])
#将logits值转换为词元概率，后续使用multinomial 和 温度缩放 进行概率采样
topk_probas = torch.softmax(new_logits, dim=0)
# print(topk_probas)
# tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])

#修改文本生成函数
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]#获取最后一个token向量
        if top_k is not None:   #a.Top_K采样获取前K个logits，其他填充负无穷
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:   #b.温度缩放：0贪婪采样、<1采样陡峭、>1采样均匀
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx  = torch.cat((idx, idx_next), dim=1)
    return idx

torch.manual_seed(123)
# token_ids = generate(
#     model = model,
#     idx = text_to_token_ids("Never think of it", tokenizer),
#     max_new_tokens = 15,
#     context_size = GPT_CONFIG_124M["context_length"],
#     # top_k = 1,
#     # temperature = 0
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# 5.4使用pytorch保存模型权重
# torch.save(model.state_dict(), 'model.pth')
# #加载权重
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(torch.load('model.pth', map_location=device))
# model.eval() # 推断模式，禁用模型的dropout层，不丢弃网络学习到的任何信息
# #保存模型和优化的stat_dict内容
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     },
#     'model_and_optimizer.pth'
# )
# #加载和恢复模型和优化的状态
# checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# model.train()


# 5.5从OpenAI加载预训练权重
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

import torch
def load_gpt2_weights_into_model(model, model_path):
    """
    将 GPT-2 模型权重加载到自定义模型中

    Args:
        model: 自定义模型实例
        model_path: GPT-2 模型路径
    """
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        # 直接复制数据到现有张量
        left.data.copy_(right.clone().detach())
    
    # 加载 GPT-2 模型权重
    gpt2_model = torch.load(model_path)
    
    # 1. 处理词嵌入层
    assign(model.tok_emb.weight, gpt2_model["wte.weight"])
    
    # 2. 处理位置嵌入层
    assign(model.pos_emb.weight, gpt2_model["wpe.weight"])
    
    # 3. 处理 Transformer 块
    for layer_idx in range(len(model.trf_blocks)):
        # 3.1 第一个层归一化
        assign(model.trf_blocks[layer_idx].norm1.scale, gpt2_model[f"h.{layer_idx}.ln_1.weight"])
        assign(model.trf_blocks[layer_idx].norm1.shift, gpt2_model[f"h.{layer_idx}.ln_1.bias"])
        
        # 3.2 多头自注意力层
        # 拆分 QKV 权重
        qkv_weight = gpt2_model[f"h.{layer_idx}.attn.c_attn.weight"]
        q_weight, k_weight, v_weight = torch.split(qkv_weight, qkv_weight.size(1)//3, dim=-1)
        assign(model.trf_blocks[layer_idx].attn.W_query.weight, q_weight.T)
        assign(model.trf_blocks[layer_idx].attn.W_key.weight, k_weight.T)
        assign(model.trf_blocks[layer_idx].attn.W_value.weight, v_weight.T)
        
        # 拆分 QKV 偏置
        qkv_bias = gpt2_model[f"h.{layer_idx}.attn.c_attn.bias"]
        q_bias, k_bias, v_bias = torch.split(qkv_bias, qkv_bias.size(0)//3, dim=-1)
        assign(model.trf_blocks[layer_idx].attn.W_query.bias, q_bias)
        assign(model.trf_blocks[layer_idx].attn.W_key.bias, k_bias)
        assign(model.trf_blocks[layer_idx].attn.W_value.bias, v_bias)
        
        # 输出投影层
        assign(model.trf_blocks[layer_idx].attn.out_proj.weight, gpt2_model[f"h.{layer_idx}.attn.c_proj.weight"].T)
        assign(model.trf_blocks[layer_idx].attn.out_proj.bias, gpt2_model[f"h.{layer_idx}.attn.c_proj.bias"])

        # 3.3 第二个层归一化
        assign(model.trf_blocks[layer_idx].norm2.scale, gpt2_model[f"h.{layer_idx}.ln_2.weight"])
        assign(model.trf_blocks[layer_idx].norm2.shift, gpt2_model[f"h.{layer_idx}.ln_2.bias"])

        # 3.4 前馈神经网络
        assign(model.trf_blocks[layer_idx].ffn.layers[0].weight, gpt2_model[f"h.{layer_idx}.mlp.c_fc.weight"].T)
        assign(model.trf_blocks[layer_idx].ffn.layers[0].bias, gpt2_model[f"h.{layer_idx}.mlp.c_fc.bias"])
        assign(model.trf_blocks[layer_idx].ffn.layers[2].weight, gpt2_model[f"h.{layer_idx}.mlp.c_proj.weight"].T)
        assign(model.trf_blocks[layer_idx].ffn.layers[2].bias, gpt2_model[f"h.{layer_idx}.mlp.c_proj.bias"])

    # 4. 处理最后的层归一化
    assign(model.final_norm.scale, gpt2_model["ln_f.weight"])
    assign(model.final_norm.shift, gpt2_model["ln_f.bias"])

    # 5. 处理输出层（权重绑定）
    # GPT-2 模型在其输出层中复用了词元嵌入层的权重，以减少参数总数，这一概念称为权重绑定。
    # 因此这里我们只需将词元嵌入层的权重复制到输出层即可。
    assign(model.out_head.weight, gpt2_model["wte.weight"])

    return model

gpt = GPTModel(NEW_CONFIG)
gpt.eval()
# 加载gpt2权重
load_gpt2_weights_into_model(gpt, "pytorch_model.bin")
gpt.to(device)
torch.manual_seed(123)
token_ids = generate(
    model = gpt,
    idx = text_to_token_ids("I HAD always thought Jack", tokenizer).to(device),
    max_new_tokens = 25,
    context_size = NEW_CONFIG["context_length"],
    top_k= 25,
    temperature = 1.5
)
print("Ooutput text:\n", token_ids_to_text(token_ids, tokenizer))