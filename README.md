# MiniGPT

从0到1构建一个大语言模型LLM，从数据准备、GPT主架构、Transformer Block、基础模型预训练、分类微调、指令微调到基础文本补全、垃圾消息分类、指令遵从的实现。

## 目录结构

```
MiniGPT
├── chapters/                           # 章节笔记
│   └── ...                        
├── configs/                            # 配置文件
│   └── ...
├── data/                               # 原始数据集
│   └── ...
├── model                               # 模型架构
│   ├── __init__.py
│   ├── attention.py                    # 多头掩码注意力层
│   ├── feed_forward.py                 # 前馈神经网络层
│   ├── gpt_model.py                    # 模型主体
│   ├── layer_norm.py                   # 层归一化
│   └── transformer_block.py            # Transformer块
├── utils                               # 工具组件
│   ├── __init__.py
│   ├── dataset_loader.py               # 数据处理、数据加载
│   ├── load_gpt2_weights.py            # 映射GPT2并加载权重至自实现模型
│   ├── metrics.py                      # 模型评估指标计算方法，交叉墒损失或预测准确率等
│   ├── model_inference.py              # 模型推理方法
│   └── train_model.py                  # 模型训练、微调方法
├── weights                             # 权重目录 
│   ├── instruction_executor.pth        # 基于GPT2-355M指令微调后的权重
│   ├── pytorch_model.bin               # 124M权重
│   ├── pytorch_model_355m.bin          # 355M权重
│   └── review_classifier.pth           # 基于GPT2-124M分类微调后的权重
├── base_generate.py                    # 基础文本补全
├── base_pretrain.py                    # 预训练
├── class_finetune_inference.py         # 垃圾消息分类推理 
├── class_finetune_train.py             # 消息分类微调
├── instruction_finetune_inference.py   # 指令执行推理
├── instruction_finetune_train.py       # 指令遵从微调
├── README.md                          
├── requirements.txt
└── tree.py                             # 树状目录生成脚本
```

## 基础模型（一）

#### 预训练：

```bash
python base_pretrain.py --config configs/gpt2_config_124m_ctx256.json --data_path data/the-verdict.txt --model_path weights/model_ctx256.pth
```

**参数说明：**

| 参数           | 说明       | 必填  | 默认值                                  |
| ------------ | -------- | --- | ------------------------------------ |
| `config`     | 模型配置文件路径 | 否   | configs/gpt2_config_124m_ctx256.json |
| `data_path`  | 数据源文件路径  | 否   | data/the-verdict.txt                 |
| `model_path` | 模型保存路径   | 否   | weights/model_ctx256.pth             |

#### 文本生成：

```bash
python base_pretrain.py --config configs/gpt2_config_124m_ctx256.json --model_path weights/model_ctx256.pth --max_new_tokens 50 --temperature 1.2 --top_k 35
```

**参数说明：**

| 参数 | 说明  | 必填  | 默认值 |
| --- | --- | --- | --- |
| config | 配置文件路径 | 否 | configs/gpt2_config_124m_ctx256.json |
| model_path | 模型权重加载路径 | 否 | weights/model_ctx256.pth |
| max_new_tokens | 新生成的token最大数量 | 否 | 50 |
| temperature | 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定 | 否 | 0.0 |
| top_k | top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定 | 否 | None |

测试效果:

```bash
python3 base_generate.py
开始对话（输入'exit'退出）

用户: I HAD always thought Jack

模型: I HAD always thought Jack Gisburn rather a cheap genius--I told Mrs. Stroud so when she began to stammer something about her poverty. Gisburn's open countenance. "It's his ridiculous modesty, you know. He says they're not
```

## 分类微调（二）

#### 微调训练：

```bash
python --config configs/gpt2_config_124m.json --data_path data/SMSSpamCollection.tsv --model_path weights/review_classifier.pth --gpt2_model_path weights/pytorch_model.bin
```

**参数说明：**

| 参数  | 说明  | 必填  | 默认值 |
| --- | --- | --- | --- |
| config | 配置文件路径 | 否 | configs/gpt2_config_124m.json |
| data_path | 分类数据源文件路径 | 否 | data/SMSSpamCollection.tsv |
| model_path | 模型权重保存路径 | 否 | weights/review_classifier.pth |
| gpt2_model_path | GPT2模型加载路径 | 否 | weights/pytorch_model.bin |

#### 消息分类：

```bash 
python --config configs/gpt2_config_124m.json --model_path weights/review_classifier.pth
```

**参数说明：**

| 参数  | 说明  | 必填  | 默认值 |
| --- | --- | --- | --- |
| config | 配置文件路径 | 否 | configs/gpt2_config_124m.json |
| model_path | 分类微调后模型权重文件路径 | 否 | weights/review_classifier.pth |

测试效果:

```bash
python3 class_finetune_inference.py
开始对话（输入'exit'退出）

用户: You have won ?1,000 cash or a ?2,000 prize! To claim, call09050000327
模型: not spam

用户: WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
模型: spam
```

## 指令微调（三）

#### 微调训练：

```bash
python --data_path data/instruction-data.json --config configs/gpt2_config_355m.json --gpt2_model_path weights/pytorch_model_355m.bin --model_path weights/instruction_executor.pth
```

**参数说明：**

| 参数  | 说明  | 必填  | 默认值 |
| --- | --- | --- | --- |
| data_path | 指令微调数据集文件路径 | 否 | data/instruction-data.json |
| config | GPT2-355M配置文件路径 | 否 | configs/gpt2_config_355m.json |
| gpt2_model_path | GPT2-355M基础模型权重文件路径 | 否 | weights/pytorch_model_355m.bin |
| model_path | 指令微调的模型权重保存路径 | 否 | weights/instruction_executor.pth |

#### 指令遵从：

```bash
python --config configs/gpt2_config_355m.json --model_path weights/instruction_executor.pth --max_new_tokens 768 --temperature 0.0 --top_k 25
```

**参数说明：**

| 参数  | 说明  | 必填  | 默认值 |
| --- | --- | --- | --- |
| config | GPT2-355M配置文件路径 | 否 | configs/gpt2_config_355m.json |
| model_path | 经过指令微调后的模型权重文件路径 | 否 | weights/instruction_executor.pth |
| max_new_tokens | 新生成的token最大数量 | 否 | 768 |
| temperature | 温度，用于控制生成文本的随机性 | 否 | 0.0 |
| top_k | top-k 采样，只从概率最高的 k 个 token 中采样 | 否 | None |

测试效果:

```bash
python3 instruction_finetune_inference.py
开始对话（输入'exit'退出）

任务指令: Identify the type of sentence.
任务输入: Did you finish the report?
模型: The type of sentence is imperative.

任务指令: Name the author of 'Pride and Prejudice'.
任务输入:
模型: The author of 'Pride and Prejudice' is Jane Austen.
```

## 注意！

 1. **在使用模型推理时需要保证配置文件与模型权重的参数配置一致**

## 参考资料

[《从零构建大模型》](https://book.douban.com/subject/37305124/) 

[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

[PocketLLm](https://github.com/mcuking/PocketLLM/tree/master)

[ OpenAI GPT2-124M(small)权重文件 ](https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin)

[ OpenAI GPT2-355M(medium)权重文件 ](https://huggingface.co/openai-community/gpt2-medium/resolve/main/pytorch_model.bin)