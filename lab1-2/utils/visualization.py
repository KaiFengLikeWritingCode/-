import matplotlib.pyplot as plt


def plot_training_history(train_history, val_history, title="Training History"):
    """
    绘制训练与验证过程中的损失和准确率曲线
    参数：
        train_history: 字典，包含训练阶段 loss 和 acc
        val_history: 字典，包含验证阶段 loss 和 acc
        title: 图表标题
    """
    epochs = range(1, len(train_history['loss']) + 1)

    # 绘制损失曲线
    plt.figure()
    plt.plot(epochs, train_history['loss'], 'bo-', label='Train Loss')
    plt.plot(epochs, val_history['loss'], 'ro-', label='Val Loss')
    plt.title(title + " - Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(epochs, train_history['acc'], 'bo-', label='Train Accuracy')
    plt.plot(epochs, val_history['acc'], 'ro-', label='Val Accuracy')
    plt.title(title + " - Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
