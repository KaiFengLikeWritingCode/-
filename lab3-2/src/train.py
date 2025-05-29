#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, logging, yaml, argparse
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.data import load_sst, normalize, build_dataset
from utils.vmd  import vmd_decompose
from utils.viz  import plot_loss
from models.joint_lstm import JointLSTM

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    counter  = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs+1):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)
        train_losses.append(tr_loss)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch} | Train MSE={tr_loss:.6f} | Val MSE={val_loss:.6f}")

        # early stop
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info("Early stopping.")
                break

    return train_losses, val_losses

def main(args):
    setup_logging()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('models', exist_ok=True)
    os.makedirs('plots',  exist_ok=True)

    # 1. load & normalize
    ts, time = load_sst(cfg['input'])
    ts_norm, g_scaler = normalize(ts)

    # 2. seasonal features
    months = time.month.values
    sin_m  = np.sin(2*np.pi*(months-1)/12)
    cos_m  = np.cos(2*np.pi*(months-1)/12)

    # 3. VMD
    modes = vmd_decompose(
        ts_norm,
        cfg['k'],
        cfg['alpha'],
        cfg['tau'],
        cfg['tol']
    )

    # 4. concat features
    features = np.concatenate([
        modes,
        sin_m[:,None],
        cos_m[:,None]
    ], axis=1)

    # 5. build dataset & loaders
    X, Y = build_dataset(
        features, ts_norm,
        cfg['window'], cfg['horizon']
    )
    split = int(0.8 * len(X))
    X_tr, Y_tr = X[:split], Y[:split]
    X_va, Y_va = X[split:], Y[split:]
    train_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                              batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, Y_va),
                              batch_size=cfg['batch_size'], shuffle=False)

    # 6. model & train
    model = JointLSTM(
        in_ch  = cfg['k']+2,
        hidden = cfg['hidden'],
        layers = cfg['layers'],
        horizon= cfg['horizon'],
        dropout= cfg['dropout']
    )
    save_path = 'models/joint_lstm.pth'
    tr_losses, va_losses = train_model(
        model, train_loader, val_loader,
        device, save_path,
        cfg['max_epochs'], cfg['patience']
    )
    plot_loss(tr_losses, va_losses,
              'plots/joint_loss.png', 'Joint LSTM Loss')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args)
