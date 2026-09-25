from .metrics import calc_loss_batch, calc_loss_loader

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
        # 每轮之后打印一个文本样本
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    '''
    评估模型损失（数值）
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



# 评估模型（文本）
# def generate_and_print_sample(model, tokenizer, device, start_context):
#     model.eval()
#     context_size = model.pos_emb.weight.shape[0]
#     encode = text_to_token_ids(start_context, tokenizer).to(device)
#     with torch.no_grad():
#         token_ids = generate_text_simple(model=model, idx=encode, max_new_tokens=50, context_size=context_size)
#     decode_text = token_ids_to_text(token_ids, tokenizer)
#     print(decode_text.replace("\n", ""))
#     model.train()