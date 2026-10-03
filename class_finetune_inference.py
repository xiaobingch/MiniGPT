from model.gpt_model import GPTModel
from configs.config import GPT_CONFIG_124M
from utils.model_inference import classify_review
import torch
import tiktoken

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
# 初始化模型并加载 GPT-2 预训练权重
##############################

model = GPTModel(GPT_CONFIG_124M)
model.to(device)

# 评论分类任务，因此我们只需替换最后的输出层即可，
# 该层原本是将输入映射为 50257 维的向量，即词汇表的大小，
# 现在将其输出层作用改为映射为 2 维的向量，即 0/1 两类的分类器
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features = GPT_CONFIG_124M['emb_dim'],
    out_features = num_classes
)

# 加载之前训练好的模型权重参数，weights_only=True 表示只加载模型参数，不加载优化器等状态信息
model_path="weights/review_classifier.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))

# 切换为推理模式，将禁用 dropout 等只在训练时使用的功能
model.eval()

# 使用模型进行分类评论
text_1 = "Shop till u Drop, IS IT YOU, either 10K, 5K, £500 Cash or £100 Travel voucher, Call now, 09064011000. NTT PO Box CR01327BT fixedline Cost 150ppm mobile vary"

text_2 = "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today"

label = classify_review(
    text=text_2, 
    model=model, 
    tokenizer=tokenizer,
    device=device,
    max_length=GPT_CONFIG_124M["context_length"]
)

print(f"模型: {label}\n")