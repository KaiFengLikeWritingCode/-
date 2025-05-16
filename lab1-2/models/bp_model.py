import torch.nn as nn
import torch.nn.functional as F

class BPModel(nn.Module):
    """
    构建10层全连接（BP）网络模型，适用于扁平化后的输入图像。
    对 MNIST（28x28）和 CIFAR-10（32x32x3）均采用扁平化输入，
    可根据需求调整隐藏层神经元个数、初始化方式和激活函数等。
    """
    def __init__(self, input_dim, num_classes=10):
        super(BPModel, self).__init__()
        # 本示例构建10层网络（9个全连接隐藏层 + 1个输出层）
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 32)
        self.fc10 = nn.Linear(32, num_classes)
        self.activation = nn.ReLU()  # 可替换为其他激活函数

    def forward(self, x):
        # 将输入展平
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.activation(self.fc7(x))
        x = self.activation(self.fc8(x))
        x = self.activation(self.fc9(x))
        x = self.fc10(x)
        return x

def create_bp_model(input_shape, num_classes=10):
    """
    根据输入图片尺寸构建 BP 模型
    :param input_shape: tuple，例如 MNIST 为 (1, 28, 28)，CIFAR-10 为 (3, 32, 32)
    :param num_classes: 分类数（默认为10）
    :return: BPModel 模型实例
    """
    input_dim = 1
    for d in input_shape:
        input_dim *= d
    return BPModel(input_dim, num_classes)
