import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

class SSTDataset(Dataset):
    # def __init__(self, ds, ocean_mask):
    def __init__(self, ds, ocean_mask, T_IN, T_OUT):
        """
        ds: xarray.Dataset
        ocean_mask: DataArray (lat,lon) True 表示海点
        """
        sst = ds['analysed_sst']                          # [time,lat,lon]
        self.mask = ocean_mask                            # [lat,lon]
        sst = sst.interpolate_na('time', 'linear')
        # —— 1) 插值 & 标准化 ——
        sst = sst.interpolate_na('time', 'linear')
        μ   = float(sst.where(ocean_mask).mean(skipna=True))
        σ   = float(sst.where(ocean_mask).std(skipna=True))
        sst = (sst - μ) / σ
        # —— **关键：把所有 NaN（陆地）填成 0** ——
        sst = sst.fillna(0)

        arr = sst.values                                  # [T, H, W]
        T, H, W = arr.shape

        # —— 2) 周平均场 ——
        wkmean = np.zeros_like(arr)
        for t in range(T):
            wkmean[t] = arr[max(0, t-6):t+1].mean(axis=0)

        # —— 3) 季节编码 ——
        times       = ds['time'].values.astype('datetime64[D]')
        day_of_year = np.array([d.astype(object).timetuple().tm_yday for d in times])
        sin365      = np.sin(2*np.pi*day_of_year/365)[:,None,None]
        cos365      = np.cos(2*np.pi*day_of_year/365)[:,None,None]

        # —— 4) 纬经度通道 ——
        lats    = ds['latitude'].values
        lons    = ds['longitude'].values
        lon_map = np.broadcast_to(lons, (H,W))
        lat_map = np.broadcast_to(lats[:,None], (H,W))

        # —— 合并通道：SST, 周均, sin, cos, lat, lon ——
        all_ch = np.stack([
            arr,
            wkmean,
            np.broadcast_to(sin365, (T,H,W)),
            np.broadcast_to(cos365, (T,H,W)),
            np.broadcast_to(lat_map[None], (T,H,W)),
            np.broadcast_to(lon_map[None], (T,H,W)),
        ], axis=-1)  # [T,H,W,6]

        # —— 5) 构造滑动窗口样本 ——
        Xs, Ys = [], []
        for i in range(T - T_IN - T_OUT + 1):
            Xs.append(all_ch[i : i + T_IN])                       # [T_IN,H,W,6]
            Ys.append(arr[i+T_IN : i+T_IN+T_OUT][...,None])       # [T_OUT,H,W,1]
        self.X = np.stack(Xs)  # [N, T_IN, H, W, 6]
        self.Y = np.stack(Ys)  # [N, T_OUT, H, W, 1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        # 转成 [T, C, H, W]
        x = torch.from_numpy(self.X[idx]).permute(0,3,1,2).float()
        y = torch.from_numpy(self.Y[idx]).permute(0,3,1,2).float()
        # 掩膜转成 [H, W]
        mask = torch.from_numpy(self.mask.values).float()
        return x, y, mask
