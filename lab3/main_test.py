#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_joint_lstm.py

测试端到端多通道 Joint LSTM 模型。
1. 加载 config.yaml
2. 加载 NetCDF 数据并做 VMD 分解
3. 构建多通道滑窗验证集 X_val, Y_val
4. 加载训练好的 JointLSTM 权重
5. 滑窗做 1-step 预测，保存 plots/forecast_1step_test.png
6. 滑窗多步预测叠加，保存 plots/forecast_overlay_test.png
"""

import os
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD

# -----------------------------
def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_sst(path):
    ds = xr.open_dataset(path)
    da = ds['thetao_cglo'].sel(
        depth=ds.depth[0],
        latitude=ds.latitude[0],
        longitude=ds.longitude[0]
    )
    ts = da.values.astype('float32')
    time = pd.to_datetime(ds['time'].values)
    return ts, time

def normalize(x):
    scaler = MinMaxScaler((0,1))
    y = scaler.fit_transform(x.reshape(-1,1)).flatten()
    return y, scaler

def vmd_decompose(x, cfg):
    alpha = float(cfg['alpha'])
    tau   = float(cfg['tau'])
    K     = int(cfg['k'])
    tol   = float(cfg['tol'])
    modes, _, _ = VMD(x, alpha, tau, K, False, 1, tol)
    return np.array(modes).T  # (N, K)

def build_multi_dataset(modes, ts_norm, window, horizon):
    X, Y = [], []
    N, K = modes.shape
    for i in range(N - window - horizon + 1):
        X.append(modes[i:i+window, :])
        Y.append(ts_norm[i+window:i+window+horizon])
    X = torch.tensor(np.stack(X), dtype=torch.float32)      # (M, window, K)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)      # (M, horizon)
    return X, Y

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

# -----------------------------
if __name__ == '__main__':
    # 1. 加载配置与环境
    cfg = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    # 2. 加载数据 & 归一化
    ts, time = load_sst(cfg['input'])
    ts_norm, g_scaler = normalize(ts)

    # 3. VMD 分解 & 构建滑窗验证集
    modes = vmd_decompose(ts_norm, cfg)  # (N, K)
    X, Y = build_multi_dataset(modes, ts_norm,
                               cfg['window'], cfg['horizon'])
    M = len(X)
    split = int(0.8 * M)
    X_val = X[split:]
    Y_val = Y[split:]
    # 对应时间索引，用于画图
    times_val = time[ split + cfg['window'] : split + cfg['window'] + len(Y_val) ]

    # 4. 加载模型
    model = JointLSTM(cfg['k'], cfg['hidden'],
                      cfg['layers'], cfg['horizon'],
                      cfg['dropout'])
    model.load_state_dict(
        torch.load('models/joint_lstm.pth', map_location=device)
    )
    model.to(device).eval()

    # 5. 滑窗 1-step 预测
    preds1, trues1 = [], []
    with torch.no_grad():
        for i in range(len(X_val)):
            seq = X_val[i].unsqueeze(0).to(device)      # (1, window, K)
            y_norm = model(seq).cpu().numpy().flatten() # (horizon,)
            preds1.append(y_norm[0])                    # 第一时刻预测
            trues1.append(Y_val[i,0].item())            # 真值
    preds1 = np.array(preds1)
    trues1 = np.array(trues1)
    # 反归一化
    preds1_abs = g_scaler.inverse_transform(preds1.reshape(-1,1)).flatten()
    trues1_abs = g_scaler.inverse_transform(trues1.reshape(-1,1)).flatten()

    # 绘制 1-step 预测
    plt.figure(figsize=(10,4))
    plt.plot(time, ts, color='C0', alpha=0.3, label='Original')
    plt.plot(times_val, preds1_abs, color='C1', label='1-step Forecast')
    plt.xlabel('Time'); plt.ylabel('Sea Surface Temperature')
    plt.title('1-step Joint LSTM Forecast')
    plt.legend(); plt.tight_layout()
    plt.savefig('plots/forecast_1step_test.png', dpi=300)
    plt.close()

    # 6. 滑窗多步预测叠加
    plt.figure(figsize=(10,4))
    plt.plot(time, ts, color='C0', alpha=0.3, label='Original')
    with torch.no_grad():
        for i in range(len(X_val)):
            seq = X_val[i].unsqueeze(0).to(device)
            y_norm = model(seq).cpu().numpy().flatten()  # (horizon,)
            y_abs  = g_scaler.inverse_transform(y_norm.reshape(-1,1)).flatten()
            start = times_val[i]
            dates = [start + pd.Timedelta(days=j) for j in range(cfg['horizon'])]
            plt.plot(dates, y_abs, color='C1', alpha=0.15)
    plt.xlabel('Time'); plt.ylabel('Sea Surface Temperature')
    plt.title('Multi-step Joint LSTM Forecast Overlay')
    plt.tight_layout()
    plt.savefig('plots/forecast_overlay_test.png', dpi=300)
    plt.close()

    print("Test forecasts saved under plots/:")
    print(" - 1-step:        plots/forecast_1step_test.png")
    print(" - multi-step:    plots/forecast_overlay_test.png")
