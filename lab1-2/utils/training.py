import torch

def train_model(model, train_loader, val_loader, epochs, device, criterion, optimizer):
    """
    对单一折（fold）数据进行模型训练和验证
    参数：
        model: 待训练的模型
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        epochs: 训练周期数
        device: 训练设备（CPU 或 GPU）
        criterion: 损失函数
        optimizer: 优化器
    返回：
        train_history: 包含每个 epoch 训练损失和准确率（字典）
        val_history: 包含每个 epoch 验证损失和准确率（字典）
    """
    model.to(device)
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_history['loss'].append(epoch_loss)
        train_history['acc'].append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_history['loss'].append(epoch_val_loss)
        val_history['acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f} | "
              f"Val loss: {epoch_val_loss:.4f}, Val acc: {epoch_val_acc:.4f}")
    return train_history, val_history
