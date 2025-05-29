#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, yaml
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.data import load_sst, normalize, build_dataset
from utils.vmd  import vmd_decompose
from models.joint_lstm import JointLSTM

def main(config):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    # load data
    ts, time = load_sst(cfg['input'])
    ts_norm, g_scaler = normalize(ts)

    # VMD
    modes = vmd_decompose(
        ts_norm, cfg['k'], cfg['alpha'], cfg['tau'], cfg['tol']
    )

    # seasonal
    months = time.month.values
    sin_m  = np.sin(2*np.pi*(months-1)/12)
    cos_m  = np.cos(2*np.pi*(months-1)/12)

    # features & dataset
    features = np.concatenate([modes, sin_m[:,None], cos_m[:,None]], axis=1)
    X, Y = build_dataset(features, ts_norm,
                         cfg['window'], cfg['horizon'])
    split = int(0.8 * len(X))
    X_val = X[split:]; Y_val = Y[split:]
    times1 = time[ split + cfg['window'] :
                   split + cfg['window'] + len(X_val) ]

    # load model
    model = JointLSTM(
        in_ch  = cfg['k']+2,
        hidden = cfg['hidden'],
        layers = cfg['layers'],
        horizon= cfg['horizon'],
        dropout= cfg['dropout']
    )
    state = torch.load('models/joint_lstm.pth',
                       map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # 1-step forecast
    preds1 = []
    with torch.no_grad():
        for i in range(len(X_val)):
            xb = X_val[i].unsqueeze(0).to(device)
            yh = model(xb).cpu().numpy().flatten()[0]
            preds1.append(yh)
    p1 = g_scaler.inverse_transform(
        np.array(preds1).reshape(-1,1)
    ).flatten()

    # plot 1-step
    split_time = time[ split + cfg['window'] ]
    plt.figure(figsize=(12,5))
    plt.plot(time, ts, alpha=0.4, label='real')
    plt.plot(times1, p1,   label='1-step')
    plt.axvline(split_time, color='g', ls='--')
    plt.legend(); plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('1-step Joint LSTM Forecast')
    plt.tight_layout()
    plt.savefig('plots/forecast_1step_test.png', dpi=300)
    plt.close()

    # multi-step overlay
    plt.figure(figsize=(12,5))
    plt.plot(time, ts, alpha=0.4)
    with torch.no_grad():
        for i in range(len(X_val)):
            xb = X_val[i].unsqueeze(0).to(device)
            ys = model(xb).cpu().numpy().flatten()
            ys = g_scaler.inverse_transform(ys.reshape(-1,1)).flatten()
            t0 = times1[i]
            dates = [t0 + pd.Timedelta(days=j)
                     for j in range(cfg['horizon'])]
            plt.plot(dates, ys, color='C1', alpha=0.2)
    plt.axvline(split_time, color='g', ls='--')
    plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('Multi-step Joint LSTM Overlay')
    plt.tight_layout()
    plt.savefig('plots/forecast_overlay_test.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args.config)
