import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from models.bp_model import create_bp_model
from models.cnn_model import create_cnn_model

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 10
SAVED_DIR = './saved_models'
RESULTS_DIR = './eval_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 获取测试集 DataLoader
def get_test_loader(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        ds = MNIST('./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        ds = CIFAR10('./data', train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader, ds

# 收集预测结果
def collect_predictions(model, loader):
    y_true, y_pred, y_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.append(probs)
    return np.array(y_true), np.array(y_pred), np.vstack(y_prob)

# 可视化混淆矩阵
def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, None]
    plt.figure(figsize=(6,6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{name} Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(NUM_CLASSES)
    plt.xticks(ticks, ticks, rotation=45)
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name}_confusion.png'))
    plt.close()

# 主流程：遍历模型文件评估
if __name__ == '__main__':
    results = {}
    for fname in os.listdir(SAVED_DIR):
        if not fname.endswith('.pth'): continue
        # 文件名格式： dataset_model.pth
        base = fname[:-4]
        parts = base.split('_')
        if len(parts) != 2:
            continue
        dataset_name, model_type = parts
        name = f'{dataset_name.upper()}_{model_type.upper()}'
        print(f'Evaluating {name}')
        # 载入测试集
        test_loader, _ = get_test_loader(dataset_name)
        # 构建模型
        if model_type == 'bp':
            model = create_bp_model(test_loader.dataset[0][0].shape, NUM_CLASSES)
        else:
            model = create_cnn_model(test_loader.dataset[0][0].shape, NUM_CLASSES)
        # 加载参数
        model.load_state_dict(torch.load(os.path.join(SAVED_DIR, fname), map_location=DEVICE))
        model.to(DEVICE)
        # 预测
        y_true, y_pred, _ = collect_predictions(model, test_loader)
        # 计算指标
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
        results[name] = {'accuracy': acc, 'report': report}
        # 输出并保存混淆矩阵
        print(f'Accuracy: {acc:.4f}')
        print(report)
        plot_confusion(y_true, y_pred, name)
    # 汇总结果文件
    with open(os.path.join(RESULTS_DIR, 'summary.txt'), 'w') as f:
        for name, info in results.items():
            f.write(f'{name} Accuracy: {info["accuracy"]:.4f}\n')
            f.write(info['report'] + '\n')
    print('Evaluation finished. Results saved in', RESULTS_DIR)
