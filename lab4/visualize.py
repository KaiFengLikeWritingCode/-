# visualize_test_predictions.py

import os
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 从 configs.py 导入所有超参
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
    return ds, test_loader

def run_model_on_test(test_loader):
    model = SSTConvLSTM(in_ch=6, hid_chs=[64,128], T_OUT=T_OUT).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_v0.pth"), map_location=DEVICE)
    )
    model.eval()

    all_preds = []
    all_trues = []
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()   # [B, T_OUT, 1, H, W]
            all_preds.append(pred)
            all_trues.append(y.numpy())
    preds = np.concatenate(all_preds, axis=0)  # [N, T_OUT,1,H,W]
    trues = np.concatenate(all_trues, axis=0)
    return preds, trues

def plot_comparison(ds, preds, trues, sample_idx=0, time_idx=0, output="comparison_error_map.png"):
    # extract lat/lon
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # select one sample & time step
    true_map = trues[sample_idx, time_idx, 0, :, :]   # [H, W]
    pred_map = preds[sample_idx, time_idx, 0, :, :]
    err_map  = pred_map - true_map

    # plot side by side
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.pcolormesh(lons, lats, true_map, shading='auto')
    plt.title(f"True SST (sample={sample_idx}, day={time_idx})")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.colorbar(label="Normalized SST")

    plt.subplot(1, 3, 2)
    plt.pcolormesh(lons, lats, pred_map, shading='auto')
    plt.title("Predicted SST")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.colorbar(label="Normalized SST")

    plt.subplot(1, 3, 3)
    cmap = plt.get_cmap("coolwarm")
    plt.pcolormesh(lons, lats, err_map, cmap=cmap, shading='auto')
    plt.title("Error (Pred – True)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.colorbar(label="Error")

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved comparison figure to {output}")

if __name__ == "__main__":
    ds, test_loader = load_test_data()
    preds, trues   = run_model_on_test(test_loader)
    plot_comparison(ds, preds, trues, sample_idx=0, time_idx=0)
