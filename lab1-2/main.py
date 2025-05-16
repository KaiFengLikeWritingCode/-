import os
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from utils import data_preprocessing, training, evaluation, visualization
from models import bp_model, cnn_model


def run_experiment(dataset_name='mnist', model_type='bp', epochs=10, batch_size=32, save_dir='./saved_models'):
    """
    运行单次实验，采用指定数据集和模型类型进行训练、验证与测试，训练后保存模型。
    参数：
        dataset_name: 'mnist' 或 'cifar10'
        model_type: 'bp' 或 'cnn'
        epochs: 每折训练周期数
        batch_size: 批处理大小
        save_dir: 模型保存目录
    返回：
        test_loss, test_acc, model_path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 数据加载与预处理
    if dataset_name.lower() == 'mnist':
        train_dataset, test_dataset = data_preprocessing.load_and_preprocess_mnist()
    elif dataset_name.lower() == 'cifar10':
        train_dataset, test_dataset = data_preprocessing.load_and_preprocess_cifar10()
    else:
        raise ValueError("不支持的数据集！请选择 'mnist' 或 'cifar10'")

    # 获取输入尺寸
    input_shape = train_dataset[0][0].shape

    # 选择模型构建函数
    if model_type.lower() == 'bp':
        model_fn = bp_model.create_bp_model
    elif model_type.lower() == 'cnn':
        model_fn = cnn_model.create_cnn_model
    else:
        raise ValueError("不支持的模型类型！请选择 'bp' 或 'cnn'")

    criterion = nn.CrossEntropyLoss()
    folds = data_preprocessing.kfold_split(train_dataset, n_splits=5)
    fold_histories = []

    print(f"===== {dataset_name.upper()} - {model_type.upper()} 模型训练 =====")
    # 逐折训练
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n----- Fold {fold + 1} / {len(folds)} -----")
        model = model_fn(input_shape, num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        train_hist, val_hist = training.train_model(
            model, train_loader, val_loader,
            epochs=epochs, device=device,
            criterion=criterion, optimizer=optimizer
        )
        fold_histories.append((train_hist, val_hist))

    # 保存训练好的模型（使用最后一折模型）
    model_filename = f"{dataset_name}_{model_type}.pth"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

    # 测试评估
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("\n----- 测试评估 -----")
    test_loss, test_acc = evaluation.evaluate_model(model, test_loader, device, criterion)

    # 可视化第一折训练过程
    visualization.plot_training_history(
        fold_histories[0][0], fold_histories[0][1],
        title=f"{dataset_name.upper()} {model_type.upper()} Training History"
    )

    return test_loss, test_acc, model_path


if __name__ == '__main__':
    # 实验配置列表
    experiments = [
        ('mnist', 'bp'),
        ('mnist', 'cnn'),
        ('cifar10', 'bp'),
        ('cifar10', 'cnn'),
    ]

    results = {}
    for ds, mt in experiments:
        print("\n==============================================")
        print(f"实验：数据集 = {ds.upper()}，模型 = {mt.upper()}")
        test_loss, test_acc, model_path = run_experiment(
            dataset_name=ds,
            model_type=mt,
            epochs=5,
            batch_size=32,
            save_dir='./saved_models'
        )
        results[f"{ds}_{mt}"] = {'loss': test_loss, 'acc': test_acc, 'path': model_path}

    # 实验结果汇总
    print("\n===== 实验结果汇总 =====")
    for key, info in results.items():
        print(f"{key}: Test Loss = {info['loss']:.4f}, Test Accuracy = {info['acc']:.4f}, Model Path = {info['path']}")
