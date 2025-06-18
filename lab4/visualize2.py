# visualize_test_predictions.py

import os
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs import DATA_PATH, MODEL_DIR, DEVICE, T_IN, T_OUT, BATCH_SIZE
from data.sst_dataset import SSTDataset
from models.sst_conv_lstm import SSTConvLSTM

def load_test_data():
    ds = xr.open_dataset(DATA_PATH)
    ocean_mask = ~ds['analysed_sst'].isnull().all(dim='time')
    dataset = SSTDataset(ds, ocean_mask, T_IN, T_OUT)
    N = len(dataset)
    n_train = int(N * 0.7)
    n_val   = int(N * 0.15)
    n_test  = N - n_train - n_val
    _, _, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return ds, dataset.mask, test_loader

def run_model_on_test(test_loader):
    model = SSTConvLSTM(in_ch=6, hid_chs=[64,128], T_OUT=T_OUT).to(DEVICE)
    # 安全加载模型，仅载入权重
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "best_v2.pth"),
            map_location=DEVICE,
            weights_only=True      # <-- 新增
        )
    )
    model.eval()

    all_preds, all_trues, all_masks = [], [], []
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()    # [B, T_OUT,1,H,W]
            all_preds.append(pred)
            all_trues.append(y.numpy())
            all_masks.append(mask.numpy())
    preds = np.concatenate(all_preds, axis=0)  # [N, T_OUT,1,H,W]
    trues = np.concatenate(all_trues, axis=0)
    masks = np.concatenate(all_masks, axis=0)  # [N, H, W]
    return preds, trues, masks

def plot_comparison(ds, preds, trues, sample_idx=0, time_idx=0, output="comparison_error_map.png"):
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    true_map = trues[sample_idx, time_idx, 0, :, :]
    pred_map = preds[sample_idx, time_idx, 0, :, :]
    err_map  = pred_map - true_map

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.pcolormesh(lons, lats, true_map, shading='auto')
    plt.title("True SST")
    plt.colorbar(label="Normalized SST")
    plt.subplot(1, 3, 2)
    plt.pcolormesh(lons, lats, pred_map, shading='auto')
    plt.title("Predicted SST")
    plt.colorbar(label="Normalized SST")
    plt.subplot(1, 3, 3)
    plt.pcolormesh(lons, lats, err_map, cmap="coolwarm", shading='auto')
    plt.title("Error (Pred – True)")
    plt.colorbar(label="Error")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

def plot_error_histogram(preds, trues, masks, output="error_histogram.png"):
    # 只统计海区误差
    errors = (preds - trues).reshape(-1)
    valid = masks.repeat(preds.shape[1], axis=0).reshape(-1) == 1
    errs = errors[valid]
    plt.figure()
    plt.hist(errs, bins=50, density=True, alpha=0.7)
    plt.title("Error Histogram (Sea points)")
    plt.xlabel("Prediction Error"); plt.ylabel("Density")
    plt.savefig(output, dpi=300)
    plt.close()

def plot_scatter(preds, trues, masks, output="scatter_true_pred.png"):
    # 散点图：真值 vs 预测
    y = trues.reshape(-1)
    y_hat = preds.reshape(-1)
    valid = np.repeat(masks, preds.shape[1], axis=0).reshape(-1) == 1
    plt.figure(figsize=(6,6))
    plt.hist2d(y[valid], y_hat[valid], bins=100, cmap="viridis")
    plt.plot([-3,3],[-3,3], 'r--')
    plt.title("True vs Predicted SST")
    plt.xlabel("True SST"); plt.ylabel("Predicted SST")
    plt.colorbar(label="Counts")
    plt.savefig(output, dpi=300)
    plt.close()

def plot_lead_time_rmse(preds, trues, masks, output="lead_time_rmse.png"):
    # 对每个 lead time 计算 RMSE
    T = preds.shape[1]
    rmses = []
    for t in range(T):
        err2 = ((preds[:,t,0] - trues[:,t,0])**2 * masks).sum()
        rmse = np.sqrt(err2 / masks.sum() / preds.shape[0])
        rmses.append(rmse)
    plt.figure()
    plt.plot(range(1, T+1), rmses, marker='o')
    plt.title("Lead Time RMSE")
    plt.xlabel("Lead Time (days)"); plt.ylabel("RMSE")
    plt.xticks(range(1, T+1))
    plt.savefig(output, dpi=300)
    plt.close()

def plot_spatial_rmse(preds, trues, masks, ds, output="spatial_rmse.png"):
    """
    为了让 C 维度变为 [H, W]，需要同时在 N（batch）和 T（lead time）两轴求和：
      err2 = sum_{n,t} (error_{n,t,i,j}^2 * mask_{n,i,j})
    然后 count = (num_samples * num_leadtimes) * mask.sum(axis=0)
    """
    # preds/trues: [N, T_OUT, 1, H, W]   masks: [N, H, W]
    err2 = ((preds.squeeze(2) - trues.squeeze(2))**2 * masks[:,None,:,:]) \
             .sum(axis=(0,1))       # → [H, W]

    # 每个像素被海区样本覆盖的总次数 = masks.sum(axis=0) * T_OUT
    count = masks.sum(axis=0) * preds.shape[1]  # [H, W]

    # 避免除以 0
    valid = count > 0
    spatial_rmse = np.full_like(count, np.nan, dtype=float)
    spatial_rmse[valid] = np.sqrt(err2[valid] / count[valid])

    # 构造经纬度网格
    lats = ds['latitude'].values  # length H
    lons = ds['longitude'].values # length W
    Lon, Lat = np.meshgrid(lons, lats)  # both [H, W]

    # 绘图
    plt.figure(figsize=(6,5))
    pcm = plt.pcolormesh(Lon, Lat, spatial_rmse, shading='auto')
    plt.title("Spatial RMSE")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(pcm, label="RMSE")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


if __name__ == "__main__":
    ds, mask, test_loader = load_test_data()
    preds, trues, masks = run_model_on_test(test_loader)

    # 样本对比
    plot_comparison(ds, preds, trues, sample_idx=0, time_idx=0)

    # 误差直方图
    plot_error_histogram(preds, trues, masks)

    # 真值 vs 预测 散点图
    plot_scatter(preds, trues, masks)

    # 不同 lead time RMSE
    plot_lead_time_rmse(preds, trues, masks)

    # 空间 RMSE 分布
    plot_spatial_rmse(preds, trues, masks, ds)
