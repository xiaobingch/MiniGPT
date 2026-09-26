GPT_CONFIG_124M = {
    'vocab_size': 50257,        #词汇表大小
    # 'context_length': 1024,     #上下文长度
    'context_length': 256,     #上下文长度 单机预训练时使用过
    'emb_dim':768,              #嵌入维度
    'n_heads':12,               #注意力头数
    'n_layers':12,              #层数
    'drop_rate':0.1,            #dropout率
    'qkv_bias':768,             #查询-键-值偏置
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}