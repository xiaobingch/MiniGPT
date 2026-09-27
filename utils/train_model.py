from .metrics import calc_loss_batch, calc_loss_loader
from utils.model_inference import token_ids_to_text, text_to_token_ids, generate
import torch

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    '''
    模型训练
    Args:
        model: 语言模型
        train_loader: 训练数据集
        val_loader: 验证数据集
        optimizer: 优化器
        device: 决定训练模型在 CPU 还是 GPU 上运行
        num_epochs: 训练轮次
        eval_freq: 每隔多少个批次打印一次训练集和验证集损失
        eval_iter: 计算数据集损失时使用的批次数
        start_context: 文本评估样本
        tokenizer: 分词器
    '''
    # 初始化列表以跟踪损失和所见的词元
    train_losses, val_losses, track_tokens_seen = [], [], [] 
    tokens_seen, global_step = 0, -1

    # 遍历训练轮次，一轮就是完整地遍历一次训练数据集
    for epoch in range(num_epochs):
        # 切换模型为训练模式
        model.train()

        # 在每个训练轮次中遍历批次，批次数量由训练集的大小除以每个批次的大小确定
        for input_batch, target_batch in train_loader:
            # 重制上一个批次迭代中的损失梯度
            optimizer.zero_grad()

            # 计算损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # 进行反向传播来计算损失梯度
            loss.backward()

            # 使用损失梯度更新模型权重
            optimizer.step()

            # 已阅读token数量
            tokens_seen += input_batch.numel() 

            # 进行多少次参数更新
            global_step += 1 

            # 打印训练集/验证集损失，可选的评估步骤，每几步评估一次
            if global_step % eval_freq == 0: 

                # 计算模型在训练数据集和验证数据集的损失
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}):"
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        # 每轮之后打印一个文本样本以评估模型
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    '''
    评估模型损失(数值)
    Args:
        model:模型
        train_loader:训练数据集
        val_loader: 验证数据集
        device: 决定训练模型在 CPU 还是 GPU 上运行
        eval_iter: 计算数据集损失时使用的批次数
    '''
    # 评估阶段禁用dropout以产出稳定且可复现的结果
    model.eval()

    #评估阶段禁用梯度跟踪，减少计算开销也无必要
    with torch.no_grad(): 
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    # 恢复模型为训练模式
    model.train()
    return train_loss, val_loss



def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    接收初始提示文本，调用模型生成后续文本并在终端打印。
    常用于训练过程中定期采样，以直观观察模型的生成效果。

    Args:
        model : torch.nn.Module (如 GPTModel)
            待评估的大语言模型实例。
        tokenizer : Tokenizer 实例
            文本分词器，用于在字符串(String)与数字编号(Token ID)之间进行互相转换。
        device : torch.device (如 'cuda' 或 'cpu')
            张量(Tensor)计算的目标设备，指示模型和数据运行在 GPU 还是 CPU 上。
        start_context : str
            文本生成的起始提示词(Prompt)，即模型接龙续写的上文内容。
    """
    
    # 1. 将模型切换为评估模式(Evaluation Mode)
    # 作用：禁用 Dropout 和 LayerNorm 的动态更新，确保推理/评估过程稳定且可复现
    model.eval()
    
    # 2. 从位置编码矩阵中获取模型支持的最大上下文长度(Context Length / Block Size)
    # pos_emb.weight 的形状为 (context_size, emb_dim)，第 0 维代表最大序列长度
    context_size = model.pos_emb.weight.shape[0]
    
    # 3. 将输入的文本提示(Prompt)转换为 Token ID 序列，并移动到指定的计算设备(CPU 或 GPU)
    # 输入维度形状为：(1, sequence_length)
    encode = text_to_token_ids(start_context, tokenizer).to(device)
    
    # 4. 禁用梯度计算上下文(No-Grad Context)
    # 作用：在推理阶段不构建计算图，大幅节省显存并加快计算速度
    with torch.no_grad():
        # 调用文本生成逻辑，自回归地预测接下来的 50 个 token
        # 传入 context_size 以确保在生成过程中对输入序列进行裁剪，防止超出位置编码上限
        token_ids = generate(
            model=model, 
            idx=encode, 
            max_new_tokens=50, 
            context_size=context_size
        )
    
    # 5. 将生成的 Token ID 序列反解码(Decode)回人类可读的文本字符串
    decode_text = token_ids_to_text(token_ids, tokenizer)
    
    # 6. 打印解码后的文本，并去除换行符，以便在一行中更整洁地展示输出
    print(decode_text.replace("\n", ""))
    
    # 7. 将模型重新恢复为训练模式(Training Mode)
    # 作用：重新启用 Dropout 等层，避免影响后续继续进行的训练过程
    model.train()