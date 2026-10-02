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
├── base_generate.py                    # 文本生成
├── base_pretrain.py                    # 预训练
├── class_funetune_inference.py         # 垃圾消息分类推理 
├── class_funetune_train.py             # 消息分类微调
├── instruction_funetune_inference.py   # 指令执行推理
├── instruction_funetune_train.py       # 指令遵从微调
├── README.md                          
├── requirements.txt
└── tree.py                             # 树状目录脚本
```
## 安装
## 基础模型
### 预训练
### 文本生成
## 分类微调
### 微调训练
### 文本分类
## 指令微调
### 微调训练
### 指令遵从
## 参考资料