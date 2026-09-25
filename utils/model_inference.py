import torch

def text_to_token_ids(text, tokenizer):
    '''
    文本转tonken id
    text:输入文本
    tokenizer:分词器
    '''
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
    # squeeze 方法用于在张量维度上减少一个维度，这里在第一个维度上减少一个维度，使得输出的形状为 (num_tokens)
    # 这里使用 squeeze 方法是因为模型输出的形状为 (batch_size, num_tokens)，而我们只需要一个文本，因此需要减少一个维度
    flat = token_ids.squeeze(0)#移除batch维度
    return tokenizer.decode(flat.tolist())