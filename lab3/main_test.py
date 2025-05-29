#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_joint_lstm.py

加载训练好的 JointLSTM，做滑窗 1-step & 多步叠加预测，并可视化：
  - 原始全序列（蓝）
  - 1-step 连续预测（红）
  - 训练/验证分割线（绿虚线）
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

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    cfg    = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    # 1. 加载 & 归一化
    ts, time = load_sst(cfg['input'])
    ts_norm, g_scaler = normalize(ts)

    # 2. VMD 分解 & 构造滑窗
    modes = vmd_decompose(ts_norm, cfg)  # (N, K)
    # X, Y  = build_multi_dataset(modes, ts_norm,
    #                             cfg['window'], cfg['horizon'])

    # -- 新增：生成周期特征 sin/cos
    months = time.month.values
    sin_m = np.sin(2 * np.pi * (months - 1) / 12)[:, None]  # (N,1)
    cos_m = np.cos(2 * np.pi * (months - 1) / 12)[:, None]  # (N,1)

    # 把 VMD 模态和周期特征拼成 (N, K+2)
    features = np.concatenate([modes, sin_m, cos_m], axis=1)

    # 现在用 features（K+2 通道）来构建滑窗
    X, Y = build_multi_dataset(features, ts_norm,
                                +                           cfg['window'], cfg['horizon'])



    M     = len(X)
    split = int(0.8 * M)
    X_val = X[split:]
    Y_val = Y[split:]
    # 滑窗对应的时刻（第一步时间点）
    times1 = time[ split + cfg['window'] : split + cfg['window'] + len(X_val) ]

    # 3. 加载模型
    # model = JointLSTM(cfg['k'], cfg['hidden'],
    #                   cfg['layers'], cfg['horizon'],
    #                   cfg['dropout'])
    in_ch = cfg['k'] + 2
    model = JointLSTM(in_ch, cfg['hidden'],
                      cfg['layers'], cfg['horizon'],
                      cfg['dropout'])
    # model.load_state_dict(
    #     torch.load('models/joint_lstm.pth', map_location=device)
    # )
    state = torch.load('models/joint_lstm.pth',
                        map_location = device,
                        weights_only = True)
    model.load_state_dict(state)


    model.to(device).eval()

    # 4. 滑窗 1-step 预测
    preds1 = []
    with torch.no_grad():
        for i in range(len(X_val)):
            seq = X_val[i].unsqueeze(0).to(device)      # (1, window, K)
            yh  = model(seq).cpu().numpy().flatten()    # (horizon,)
            preds1.append(yh[0])                        # 1-step
    preds1 = np.array(preds1)
    preds1_abs = g_scaler.inverse_transform(
        preds1.reshape(-1,1)
    ).flatten()

    # 5. 画图：原始 + 1-step + 分界线
    plt.figure(figsize=(12,5))
    plt.plot(time, ts, color='C0', alpha=0.4, label='real')
    plt.plot(times1, preds1_abs, color='C1', lw=1, label='prediction')
    # 训练/验证分界
    split_time = time[ split + cfg['window'] ]
    plt.axvline(split_time, color='g', linestyle='--', lw=1.5)
    plt.legend(); plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('1-step Joint LSTM Forecast')
    plt.tight_layout()
    plt.savefig('plots/forecast_1step_test2.png', dpi=300)
    plt.close()

    # 6. 滑窗多步自回归叠加
    plt.figure(figsize=(12,5))
    plt.plot(time, ts, color='C0', alpha=0.4)
    with torch.no_grad():
        for i in range(len(X_val)):
            seq = X_val[i].unsqueeze(0).to(device)
            y_seq = model(seq).cpu().numpy().flatten()  # (horizon,)
            y_abs = g_scaler.inverse_transform(
                y_seq.reshape(-1,1)
            ).flatten()
            t0 = times1[i]
            dates = [t0 + pd.Timedelta(days=j)
                     for j in range(cfg['horizon'])]
            plt.plot(dates, y_abs, color='C1', alpha=0.2)
    plt.axvline(split_time, color='g', linestyle='--', lw=1.5)
    plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('Multi-step Joint LSTM Forecast Overlay')
    plt.tight_layout()
    plt.savefig('plots/forecast_overlay_test2.png', dpi=300)
    plt.close()

    print("Saved:")
    print(" - plots/forecast_1step_test.png")
    print(" - plots/forecast_overlay_test.png")
