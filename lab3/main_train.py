#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_train.py

端到端多通道联合建模 + 滑窗评估
- VMD → JointLSTM
- 验证集滑窗多步预测
- 1-step & 多步叠加 可视化
"""
import os, logging, yaml
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch, torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD

# --- 基本工具 ---
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def load_config(path='config.yaml'):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)
def load_sst(path):
    ds = xr.open_dataset(path)
    da = ds['thetao_cglo'].sel(depth=ds.depth[0],
                               latitude=ds.latitude[0],
                               longitude=ds.longitude[0])
    return da.values.astype('float32'), pd.to_datetime(ds['time'].values)
def normalize(x):
    scaler = MinMaxScaler((0,1))
    y = scaler.fit_transform(x.reshape(-1,1)).flatten()
    return y, scaler

def vmd_decompose(x, cfg):
    alpha,tau,K,tol = float(cfg['alpha']), float(cfg['tau']), int(cfg['k']), float(cfg['tol'])
    modes,_,_ = VMD(x, alpha, tau, K, False, 1, tol)
    return np.array(modes).T   # (N, K)

def build_multi_dataset(modes, ts_norm, window, horizon):
    X,Y = [],[]
    N,K = modes.shape
    for i in range(N-window-horizon+1):
        X.append(modes[i:i+window])
        Y.append(ts_norm[i+window:i+window+horizon])
    return (torch.tensor(np.stack(X),dtype=torch.float32),
            torch.tensor(np.stack(Y),dtype=torch.float32))

class JointLSTM(nn.Module):
    def __init__(self, in_ch, hid, layers, hor, drop):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hid, layers,
                            batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(hid, hor)
    def forward(self,x):
        o,_ = self.lstm(x)
        return self.fc(self.drop(o[:,-1,:]))

# --- 训练函数 ---
def train_model(model, X_tr,Y_tr,X_val,Y_val,device,
                save_path, max_ep,pat):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    best,val_counter = float('inf'),0
    tr_losses,val_losses = [],[]
    for ep in tqdm(range(1,max_ep+1), desc='Train',leave=False):
        model.train()
        opt.zero_grad()
        out = model(X_tr.to(device))
        l = loss_fn(out, Y_tr.to(device))
        l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(X_val.to(device))
            vl = loss_fn(vout, Y_val.to(device))
        tr_losses.append(l.item()); val_losses.append(vl.item())
        if vl<best:
            best, val_counter = vl,0
            torch.save(model.state_dict(), save_path)
        else:
            val_counter+=1
            if val_counter>=pat: break
    return tr_losses,val_losses

# --- 可视化辅助 ---
def plot_loss(tr,val,out,fname):
    plt.figure(figsize=(6,4))
    plt.plot(tr,label='Train'); plt.plot(val,label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title(fname)
    plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()

if __name__=='__main__':
    setup_logging()
    cfg = load_config()
    set_seed(cfg['seed'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('models',exist_ok=True); os.makedirs('plots',exist_ok=True)

    # 1. 数据
    ts, time = load_sst(cfg['input'])
    ts_norm, g_scaler = normalize(ts)
    modes = vmd_decompose(ts_norm, cfg)  # (N,K)

    # 2. 滑窗构建
    X, Y = build_multi_dataset(modes, ts_norm,
                               cfg['window'], cfg['horizon'])
    n = len(X); split=int(0.8*n)
    X_tr,Y_tr = X[:split],Y[:split]
    X_val,Y_val = X[split:],Y[split:]

    # 3. 联合 LSTM 训练
    model = JointLSTM(cfg['k'], cfg['hidden'],
                      cfg['layers'], cfg['horizon'], cfg['dropout'])
    mpath = 'models/joint_lstm.pth'
    tr,va = train_model(model, X_tr,Y_tr,X_val,Y_val,
                        dev,mpath,cfg['max_epochs'],cfg['patience'])
    plot_loss(tr,va,'plots/joint_loss.png','Joint LSTM Loss')

    # 4. 滑窗 1-step 预测评估
    model.load_state_dict(torch.load(mpath,map_location=dev))
    model.eval()
    preds1, truths, times1 = [],[],[]
    with torch.no_grad():
        for i in range(len(X_val)):
            seq = X_val[i].unsqueeze(0).to(dev)
            p = model(seq).cpu().numpy().flatten()[0]  # 1-step
            preds1.append(p)
            truths.append(Y_val[i,0].item())
            times1.append(time[split+i+cfg['window']])
    p1 = g_scaler.inverse_transform(np.array(preds1).reshape(-1,1)).flatten()
    t1 = g_scaler.inverse_transform(np.array(truths).reshape(-1,1)).flatten()

    plt.figure(figsize=(10,4))
    plt.plot(time,ts,color='C0',alpha=0.3,label='Original')
    plt.plot(times1, p1,'C1',label='1-step Forecast')
    plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('1-step Joint LSTM Forecast'); plt.legend()
    plt.tight_layout(); plt.savefig('plots/forecast_1step.png',dpi=300); plt.close()

    # 5. 滑窗多步叠加 overlay
    plt.figure(figsize=(10,4))
    plt.plot(time,ts,'C0',alpha=0.3,label='Original')
    with torch.no_grad():
        for i in range(len(X_val)):
            seq = X_val[i].unsqueeze(0).to(dev)
            seq_pred = model(seq).cpu().numpy().flatten()
            dates = [times1[i]+pd.Timedelta(days=j)
                     for j in range(cfg['horizon'])]
            plt.plot(dates,
                     g_scaler.inverse_transform(seq_pred.reshape(-1,1)).flatten(),
                     'C1',alpha=0.15)
    plt.xlabel('Time'); plt.ylabel('SST')
    plt.title('Multi-step Joint LSTM Overlay')
    plt.tight_layout(); plt.savefig('plots/forecast_overlay.png',dpi=300); plt.close()

    logging.info('Done. Plots in plots/, model in models/')


