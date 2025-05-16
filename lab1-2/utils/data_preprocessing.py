import numpy as np
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

def load_and_preprocess_mnist(root='./data'):
    """
    加载 MNIST 数据集，采用 ToTensor 转换（归一化到[0,1]）
    返回：train_dataset, test_dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_and_preprocess_cifar10(root='./data'):
    """
    加载 CIFAR-10 数据集，采用 ToTensor 转换（归一化到[0,1]）
    返回：train_dataset, test_dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def kfold_split(dataset, n_splits=5, random_state=42):
    """
    对数据集（索引）进行 k-fold 划分，返回每折 (train_idx, val_idx) 列表
    """
    indices = np.arange(len(dataset))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(indices))
