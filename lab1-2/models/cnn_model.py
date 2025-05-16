import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    构建3层卷积神经网络模型。
    本示例包含3个卷积层，2次最大池化操作，再接1个全连接层进行分类。
    用户可根据需要调整卷积核个数、初始化方式和激活函数等。
    """
    def __init__(self, input_channels, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层在 create_cnn_model 中根据实际输入尺寸自动定义
        self.fc = None
        self.num_classes = num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 第一次池化
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 第二次池化
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        if self.fc is None:
            raise ValueError("全连接层未定义，请使用 create_cnn_model() 创建模型")
        x = self.fc(x)
        return x

def create_cnn_model(input_shape, num_classes=10):
    """
    根据输入图片尺寸构建 CNN 模型，自动计算全连接层输入大小。
    :param input_shape: tuple，例如 MNIST 为 (1, 28, 28)，CIFAR-10 为 (3, 32, 32)
    :param num_classes: 分类数
    :return: CNNModel 模型实例
    """
    input_channels = input_shape[0]
    model = CNNModel(input_channels, num_classes)
    # 利用一个虚拟输入计算卷积输出尺寸
    dummy = torch.zeros(1, *input_shape)
    out = model.conv1(dummy)
    out = model.pool(out)
    out = model.conv2(out)
    out = model.pool(out)
    out = model.conv3(out)
    fc_input_dim = out.view(1, -1).size(1)
    model.fc = nn.Linear(fc_input_dim, num_classes)
    return model
