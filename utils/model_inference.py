import torch

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    文本生成函数
    Args:
        model: 语言模型
        idx: 由用户输入的文本转换为的tokenid, 形状(batch_size, num_tokens)
        # tokenizer: 分词器
        max_new_tokens: 新生成的 token 最大数量
        context_size: 最大上下文长度
        temperature: 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定
        top_k: top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定
    """
    for _ in range(max_new_tokens):
        # 将当前文本截断至支持的长度。如果大模型仅支持5个词元，如果输入文本长度为10，则只有最后5个词元会被用于输入文本
        idx_cond = idx[:, -context_size:]

        # no_grad 用于禁用梯度计算，只有在训练时才需要计算梯度来减小损失函数，禁用后可以加速计算减少内存占用
        with torch.no_grad():
            logits = model(idx_cond)

        # 因为模型会为每个 token 生成一个 logits，而我们只需要最后一个 token 的 logits，所以需要将维度减少一个维度，
        # 使得形状从 (batch_size, num_tokens, vocab_size) 变为 (batch_size, vocab_size)
        logits = logits[:, -1, :]#获取最后一个token向量

        if top_k is not None:   #a.Top_K采样获取前K个logits，其他填充负无穷
            # 先取出概率最高的 top_k 个 token
            top_logits, _ = torch.topk(logits, top_k)

            # 然后取这些 token 中最小的概率值
            min_val = top_logits[:, -1]

            # 将小于这个概率值的 token 的概率置为负无穷，下面 softmax/argmax 时这些 token 的概率会变为 0
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:   #b.温度缩放：0贪婪采样、<1采样陡峭、>1采样均匀
            # 可以理解为当 temperature 越小，被除之后之前的概率差距会越大，越容易生成确定性的文本
            # 而当 temperature 越大，被除之后之前的概率差距会越小，越容易生成随机性的文本
            logits = logits / temperature

            # 将 logits 分数转换为概率分布，不会改变输入顺序
            probas = torch.softmax(logits, dim=-1)

            # 使用 torch.multinomial 方法从概率分布中采样，返回形状为 (batch_size, 1) 的张量，表示每个样本的 token id
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            # 当禁用温度缩放时，就采用贪心解码，即选择概率最大的 token id 作为下一个 token
            # argmax 方法用于返回张量中每个元素的最大值的索引，
            # 这里返回的是概率最大的词元的 token id，形状为 (batch_size, 1)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # 遇到结束符，终止循环
        if idx_next == eos_id:
            break
            
        # 将新生成的 token id 添加到文本末尾，继续下一个循环，生成下一个 token
        idx  = torch.cat((idx, idx_next), dim=1)
    return idx


def text_to_token_ids(text, tokenizer):
    '''
    文本转tonken id
        text:输入文本
        tokenizer:分词器
    '''
    # 将用户输入的文本转换为 token id
    encode = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    # unsqueeze 方法用于在张量维度上增加一个维度，这里在第一个维度上增加一个维度，使得输入的形状为 (1, num_tokens)
    # 这里使用 unsqueeze 方法是因为模型要求输入的形状为 (batch_size, num_tokens)
    encode_tensor = torch.tensor(encode).unsqueeze(0)#增加batch维度
    return encode_tensor

def token_ids_to_text(token_ids, tokenizer):
    '''
    将 token id 转换为文本
        token_ids:输入文本
        tokenizer:分词器
    '''
    # # 将 token id 转换为文本
    # squeeze 方法用于在张量维度上减少一个维度，这里在第一个维度上减少一个维度，使得输出的形状为 (num_tokens)
    # 这里使用 squeeze 方法是因为模型输出的形状为 (batch_size, num_tokens)，而我们只需要一个文本，因此需要减少一个维度
    flat = token_ids.squeeze(0)#移除batch维度
    return tokenizer.decode(flat.tolist())


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    分类评论
    Args:
        text: 用户输入的文本
        model: 语言模型
        tokenizer: 分词器
        device: 决定模型在 CPU 还是 GPU 上运行
        max_length: 输入文本的最大长度，根据你的模型不同会限制最大能接受的输入长度
        pad_token_id: 填充 token 的 id, 默认使用 GPT-2 的 token id 50256 即 <|endoftext|> 来填充
    """

    # 将用户输入的文本转换为 token id
    # unsqueeze 方法用于在张量维度上增加一个维度，这里在第一个维度上增加一个维度，使得输入的形状为 (1, num_tokens)
    # 这里使用 unsqueeze 方法是因为模型要求输入的形状为 (batch_size, num_tokens)
    input_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    # 将当前文本截断至支持的长度。如果大模型仅支持5个词元，如果输入文本长度为10，则只有最后5个词元会被用于输入文本
    input_ids = input_ids[-max_length:]


    # 如果输入文本长度小于 max_length，则使用 pad_token_id 填充到 max_length
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    # 添加批次维度
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 推理时不需要计算梯度
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :] # 最后一个输出词元的logits

    # 将 logits 转换为概率分布，并使用了 argmax 方法找出概率最大的类别标签
    predicted_label = torch.argmax(logits, dim=-1).item()

    # 返回预测的类别标签，0 表示 "not spam"，1 表示 "spam"
    return "spam" if predicted_label == 1 else "not spam"