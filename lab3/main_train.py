#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_train.py

端到端多通道联合建模 + 滑窗评估
- VMD → JointLSTM
- Mini-batch 训练 + LR调度 + 权重衰减 + 梯度裁剪
- 验证集滑窗多步预测：1-step & Multi-step overlay
"""
import os, logging, yaml
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD

# --- 基本工具 ---
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path='config.yaml'):
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_sst(path):
    ds = xr.open_dataset(path)
    da = ds['thetao_cglo'].sel(depth=ds.depth[0],
                               latitude=ds.latitude[0],
                               longitude=ds.longitude[0])
    ts = da.values.astype('float32')
    time = pd.to_datetime(ds['time'].values)
    return ts, time

def normalize(x):
    scaler = MinMaxScaler((0,1))
    y = scaler.fit_transform(x.reshape(-1,1)).flatten()
    return y, scaler

def vmd_decompose(x, cfg):
    alpha,tau,K,tol = (float(cfg['alpha']), float(cfg['tau']),
                      int(cfg['k']), float(cfg['tol']))
    modes,_,_ = VMD(x, alpha, tau, K, False, 1, tol)
    return np.array(modes).T   # (N, K)

# --- 数据集构建 ---
def build_dataset(features, target, window, horizon):
    """
    features: np.ndarray (N, C)
    target:   np.ndarray (N,)
    """
    X, Y = [], []
    N, C = features.shape
    for i in range(N - window - horizon + 1):
        X.append(features[i:i+window])
        Y.append(target[i+window:i+window+horizon])
    X = torch.tensor(np.stack(X), dtype=torch.float32)  # (M, window, C)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)  # (M, horizon)
    return X, Y

# --- 模型 ---
class JointLSTM(nn.Module):
    def __init__(self, in_ch, hid, layers, hor, drop):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hid, layers,
                            batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(hid, hor)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last))

# --- 训练函数 ---
def train_model(model, train_loader, val_loader, device,
                save_path, max_epochs, patience):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs+1):
        # --- 训练 ---
        model.train()
        epoch_tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_tr_loss += loss.item() * xb.size(0)
        epoch_tr_loss /= len(train_loader.dataset)
        train_losses.append(epoch_tr_loss)

        # --- 验证 ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                epoch_val_loss += loss.item() * xb.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        logging.info(f"Epoch {epoch}: Train MSE={epoch_tr_loss:.6f}, Val MSE={epoch_val_loss:.6f}")

        # Early stopping
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info("Early stopping triggered.")
                break

    return train_losses, val_losses

# --- 可视化 ---
def plot_loss(tr, val, out_path, title):
    plt.figure(figsize=(6,4))
    plt.plot(tr, label='Train')
    plt.plot(val, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

# --- 主流程 ---
if __name__ == '__main__':
    setup_logging()
    cfg = load_config()
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # 1. 加载 & 归一化
    ts, time = load_sst(cfg['input'])
    ts_norm, g_scaler = normalize(ts)

    # 2. 周期特征
    months = time.month.values
    sin_m = np.sin(2*np.pi*(months-1)/12)
    cos_m = np.cos(2*np.pi*(months-1)/12)

    # 3. VMD 分解
    modes = vmd_decompose(ts_norm, cfg)  # (N, K)

    # 4. 拼接通道：K 模态 + 2 周期
    features = np.concatenate([modes, sin_m[:,None], cos_m[:,None]], axis=1)

    # 5. 构建滑窗数据
    X, Y = build_dataset(features, ts_norm,
                         cfg['window'], cfg['horizon'])
    M = X.size(0)
    split = int(0.8 * M)
    X_tr, Y_tr = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    # DataLoader
    batch_size = cfg.get('batch_size', 64)
    train_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val),
                              batch_size=batch_size, shuffle=False)

    # 6. 定义模型 & 训练
    model = JointLSTM(in_ch=cfg['k']+2,
                      hid=cfg['hidden'],
                      layers=cfg['layers'],
                      hor=cfg['horizon'],
                      drop=cfg['dropout'])
    mpath = 'models/joint_lstm.pth'
    tr_losses, va_losses = train_model(
        model, train_loader, val_loader, device,
        mpath, cfg['max_epochs'], cfg['patience']
    )
    plot_loss(tr_losses, va_losses,
              'plots/joint_loss.png', 'Joint LSTM Loss')

    # 7. 滑窗 1-step 预测评估
    model.load_state_dict(torch.load(mpath, map_location=device))
    model.to(device).eval()
    preds1, trues1, times1 = [], [], []
    with torch.no_grad():
        for i in range(len(X_val)):
            xb = X_val[i].unsqueeze(0).to(device)
            y_norm = model(xb).cpu().numpy().flatten()[0]
            preds1.append(y_norm)
            trues1.append(Y_val[i,0].item())
            times1.append(time[split + i + cfg['window']])
    p1 = g_scaler.inverse_transform(np.array(preds1).reshape(-1,1)).flatten()
    t1 = g_scaler.inverse_transform(np.array(trues1).reshape(-1,1)).flatten()

    plt.figure(figsize=(10,4))
    plt.plot(time, ts, color='C0', alpha=0.3, label='Original')
    plt.plot(times1, p1, color='C1', label='1-step Forecast')
    plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('1-step Joint LSTM Forecast')
    plt.legend(); plt.tight_layout()
    plt.savefig('plots/forecast_1step.png', dpi=300); plt.close()

    # 8. 滑窗多步叠加 overlay
    plt.figure(figsize=(10,4))
    plt.plot(time, ts, 'C0', alpha=0.3, label='Original')
    with torch.no_grad():
        for i in range(len(X_val)):
            xb = X_val[i].unsqueeze(0).to(device)
            y_seq = model(xb).cpu().numpy().flatten()
            dates = [times1[i] + pd.Timedelta(days=j)
                     for j in range(cfg['horizon'])]
            plt.plot(dates,
                     g_scaler.inverse_transform(y_seq.reshape(-1,1)).flatten(),
                     'C1', alpha=0.15)
    plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('Multi-step Joint LSTM Overlay')
    plt.tight_layout(); plt.savefig('plots/forecast_overlay.png', dpi=300); plt.close()

    logging.info('Training & evaluation complete.')
