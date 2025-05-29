#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMD-LSTM 多模态海表温度多步预测（加权融合 + 训练进度可视化）

配置通过 config.yaml 加载，示例内容：
```yaml
input: 'sea_temperature_water_velocity.nc'
window: 96
horizon: 16
k: 4
alpha: 2000
tau: 0
tol: 1e-6
hidden: 128
layers: 2
dropout: 0.2
seed: 42
max_epochs: 30
patience: 20
```

依赖：numpy,pandas,xarray,matplotlib,torch,scikit-learn,vmdpy,pyyaml,tqdm
"""

import logging
import warnings
import yaml
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from vmdpy import VMD

# ----------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ----------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------------------
def load_config(path: str = 'config.yaml') -> dict:
    # with open(path) as f:
    #     return yaml.safe_load(f)

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ----------------------------------------
def load_sst(nc_path: str):
    ds = xr.open_dataset(nc_path)
    da = ds['thetao_cglo'].sel(
        depth=ds.depth[0], latitude=ds.latitude[0], longitude=ds.longitude[0]
    )
    ts = da.values.astype('float32')
    time = pd.to_datetime(ds['time'].values)
    return ts, time

# ----------------------------------------
def normalize(series: np.ndarray):
    scaler = MinMaxScaler((0, 1))
    norm = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    return norm, scaler

# ----------------------------------------
def vmd_decompose(series, alpha, tau, k, dc, init, tol):
    # 确保数值类型
    alpha = float(alpha)
    tau   = float(tau)
    k     = int(k)
    init  = int(init)
    tol   = float(tol)
    modes, _, _ = VMD(series, alpha, tau, k, dc, init, tol)
    return np.array(modes)


# ----------------------------------------
def build_dataset(series: np.ndarray, window: int, horizon: int):
    X, Y = [], []
    N = len(series)
    for i in range(N - window - horizon + 1):
        X.append(series[i:i + window])
        Y.append(series[i + window:i + window + horizon])
    X = torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    return X, Y

# ----------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128,
                 num_layers=2, out_steps=16, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, out_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))

# ----------------------------------------
def train_modal(model, X_train, Y_train, X_val, Y_val,
                device, save_path, max_epochs, patience, desc):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []
    for epoch in tqdm(range(1, max_epochs + 1), desc=desc, leave=False):
        model.train()
        optimizer.zero_grad()
        out = model(X_train.to(device))
        loss = criterion(out, Y_train.to(device))
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, Y_val.to(device))
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return train_losses, val_losses

# ----------------------------------------
def predict_last(model, series_norm, scaler, window, horizon, device):
    seq = torch.tensor(series_norm[-window:], dtype=torch.float32)
    seq = seq.unsqueeze(0).unsqueeze(-1).to(device)
    model.to(device).eval()
    with torch.no_grad():
        pred_norm = model(seq).cpu().numpy().flatten()
    return scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()

# ----------------------------------------
def plot_curve(x, y1, y2, xlabel, ylabel, title, out_file):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y1, label='Train Loss')
    plt.plot(x, y2, label='Val Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

# ----------------------------------------
if __name__ == '__main__':
    setup_logging()
    cfg = load_config()
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts, time = load_sst(cfg['input'])
    ts_norm, global_scaler = normalize(ts)
    modes = vmd_decompose(ts_norm,
                          cfg['alpha'], cfg['tau'],
                          cfg['k'], False, 1, cfg['tol'])

    preds = []
    for k in range(cfg['k']):
        mode_norm, mode_scaler = normalize(modes[k])
        X, Y = build_dataset(mode_norm,
                              cfg['window'], cfg['horizon'])
        split = int(0.8 * len(X))
        X_tr, Y_tr = X[:split], Y[:split]
        X_val, Y_val = X[split:], Y[split:]
        model = LSTMForecaster(input_dim=1,
                               hidden_dim=cfg['hidden'],
                               num_layers=cfg['layers'],
                               out_steps=cfg['horizon'],
                               dropout=cfg['dropout'])
        save_path = f'modal_{k+1}.pth'
        desc = f'Modal {k+1} Training'
        tr_loss, val_loss = train_modal(
            model, X_tr, Y_tr, X_val, Y_val,
            device, save_path,
            cfg['max_epochs'], cfg['patience'], desc
        )
        plot_curve(
            list(range(1, len(tr_loss)+1)),
            tr_loss, val_loss,
            'Epoch', 'MSE', f'Modal {k+1} Loss',
            f'modal_{k+1}_loss.png'
        )
        model.load_state_dict(torch.load(save_path,
                                          map_location=device))
        pred = predict_last(
            model, mode_norm, mode_scaler,
            cfg['window'], cfg['horizon'], device
        )
        preds.append(pred)
    # 学习加权融合
    preds_arr = np.vstack(preds)
    X_meta = preds_arr.T
    y_meta = ts[-cfg['horizon']:]
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_meta, y_meta)
    w = lr.coef_
    logging.info(f'Learned fusion weights: {w}')
    final_pred = X_meta.dot(w)

    # 保存融合预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(time[-cfg['horizon']:], ts[-cfg['horizon']:], label='True')
    plt.plot(time[-cfg['horizon']:], final_pred, '--', label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Sea Surface Temperature')
    plt.title('16-step Forecast with Learned Fusion')
    plt.legend()
    plt.tight_layout()
    plt.savefig('forecast_fusion.png', dpi=300)
    logging.info('Saved forecast_fusion.png')