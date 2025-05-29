import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

def load_sst(path: str):
    """读取 NetCDF 海表温度序列和时间坐标"""
    ds = xr.open_dataset(path)
    da = ds['thetao_cglo'].sel(
        depth=ds.depth[0],
        latitude=ds.latitude[0],
        longitude=ds.longitude[0]
    )
    ts = da.values.astype('float32')
    time = pd.to_datetime(ds['time'].values)
    return ts, time

def normalize(x: np.ndarray):
    """[0,1] 归一化"""
    scaler = MinMaxScaler((0,1))
    y = scaler.fit_transform(x.reshape(-1,1)).flatten()
    return y, scaler

def build_dataset(features: np.ndarray,
                  target:    np.ndarray,
                  window:    int,
                  horizon:   int):
    """
    构造滑窗数据
    features: (N, C)
    target:   (N,)
    returns:
       X: (M, window, C)
       Y: (M, horizon)
    """
    X, Y = [], []
    N, C = features.shape
    for i in range(N - window - horizon + 1):
        X.append(features[i:i+window])
        Y.append(target[i+window:i+window+horizon])
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    return X, Y
