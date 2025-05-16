import xarray as xr
import numpy as np
import cv2
from pathlib import Path

# ---------- 读取 ----------
ds        = xr.open_dataset("glo_3ice.nc")
ssha      = ds["zos_cglo"]
sithick   = ds["sithick_cglo"]
times     = ds["time"].values

train_ssha = Path("data/train_ssha_png"); train_ssha.mkdir(exist_ok=True)
train_mask = Path("data/train_mask_png"); train_mask.mkdir(exist_ok=True)
test_ssha  = Path("data/test_ssha_png");  test_ssha.mkdir(exist_ok=True)
test_mask  = Path("data/test_mask_png");  test_mask.mkdir(exist_ok=True)

last_year_start = times[-1].astype("datetime64[Y]") - np.timedelta64(1, "Y")

# ---------- 辅助 ----------
def is_symmetric(a, thr=0.05):
    vmax, vmin = np.nanmax(a), np.nanmin(a)
    return abs(vmax + vmin) / max(abs(vmax), abs(vmin)) < thr

H = W = 512
palette = np.array([[  0,   0,   0],    # 0 sea   black
                    [  0, 150, 255],    # 1 ice   bright blue
                    [255,   0,   0]],   # 2 land  red
                   np.uint8)

# ---------- 主循环 ----------
for idx, tval in enumerate(times):
    date = np.datetime_as_string(tval, unit="D")
    ssha_slice = ssha.isel(time=idx).values
    ice_slice  = sithick.isel(time=idx).values

    ssha_dir, mask_dir = (test_ssha, test_mask) if tval >= last_year_start else (train_ssha, train_mask)

    # ---------- SSHA 灰度 ----------
    if is_symmetric(ssha_slice):
        vm   = np.nanmax(np.abs(ssha_slice))
        norm = (ssha_slice + vm) / (2*vm)
    else:
        anomaly = ssha_slice - np.nanmean(ssha_slice)
        vm      = np.nanmax(np.abs(anomaly))
        norm    = (anomaly + vm) / (2*vm)
    gray = np.uint8(np.clip(np.nan_to_num(norm)*255, 0, 255))
    gray = cv2.resize(gray, (W, H), cv2.INTER_CUBIC)
    cv2.imwrite(str(ssha_dir / f"ssha_{date}.png"), gray)

    # ---------- 先判别，后统一放大 ----------
    land_mask0 = np.isnan(ice_slice) | np.isnan(ssha_slice)
    ice_mask0  = (ice_slice > 0) & (~land_mask0)          # 只要厚度>0 即冰

    lbl0 = np.zeros_like(ssha_slice, dtype=np.uint8)
    lbl0[land_mask0]         = 2
    lbl0[ice_mask0]          = 1      # land 已经标过，海冰覆盖水面

    # 最近邻一次性放大
    lbl = cv2.resize(lbl0, (W, H), cv2.INTER_NEAREST)

    cv2.imwrite(str(mask_dir / f"lbl_{date}.png"), lbl)

    # one-hot
    oh = np.zeros((3, H, W), np.uint8)
    for c in (0,1,2):
        oh[c] = (lbl == c).astype(np.uint8)
    np.save(mask_dir / f"onehot_{date}.npy", oh)

    # 彩色调试
    cv2.imwrite(str(mask_dir / f"mask_rgb_{date}.png"), palette[lbl])
