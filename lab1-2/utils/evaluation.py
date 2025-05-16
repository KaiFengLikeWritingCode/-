import torch

def evaluate_model(model, test_loader, device, criterion):
    """
    在测试集上评估模型性能，计算平均损失和准确率
    参数：
        model: 训练好的模型
        test_loader: 测试数据 DataLoader
        device: 测试设备（CPU 或 GPU）
        criterion: 损失函数
    返回：
        avg_loss: 平均损失
        accuracy: 测试准确率
    """
    model.eval()
    model.to(device)
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            '''
             沿着类别维度（通常为最后一维）获取每个样本预测概率最大的索引。
             这里 _ 表示具体的最大值，而 predicted 表示索引，即预测的类别。
            '''
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = test_loss / total
    accuracy = correct / total
    print(f"Test loss: {avg_loss:.4f}, Test accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
