# train.py

import os
# 屏蔽 TensorFlow 的 INFO 级别日志（仅 ERROR 会显示）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 关闭 oneDNN 优化（如果你确实想禁用它）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



# -------- 1. 超参 & 配置 --------
DATA_PATH    = "cmems_obs-sst_glo_phy_nrt_l4_P1D-m_1749202248617.nc"
MODEL_DIR    = "./checkpoints"
BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS       = 15
PATIENCE     = 10
T_IN, T_OUT  = 10, 7
λ_G, λ_V     = 0.1, 0.05
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
# -------- 1. 超参 & 配置 --------
LOG_DIR = "./logs"                            # 新增：TensorBoard 日志目录
os.makedirs(LOG_DIR, exist_ok=True)

# -------- 2. ConvLSTMCell --------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hid_ch = hid_ch
        self.conv   = nn.Conv2d(in_ch + hid_ch, 4*hid_ch, kernel_size,
                                padding=padding, bias=True)

    def forward(self, x, h, c):
        # x: [B, in_ch, H, W],  h,c: [B, hid_ch, H, W]
        combined = torch.cat([x, h], dim=1)       # [B, in_ch+hid_ch, H, W]
        i, f, o, g = torch.split(self.conv(combined),
                                 self.hid_ch, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        o = torch.sigmoid(o); g = torch.tanh(g)
        c_next = f*c + i*g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# -------- 3. SSTConvLSTM 模型 --------
class SSTConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_chs=[64,128]):
        super().__init__()
        # Encoder
        self.enc_cells = nn.ModuleList([
            ConvLSTMCell(in_ch if i==0 else hid_chs[i-1], hid_chs[i])
            for i in range(len(hid_chs))
        ])
        # Decoder (镜像)
        self.dec_cells = nn.ModuleList([
            ConvLSTMCell(1 if i==0 else hid_chs[-i], hid_chs[-i-1])
            for i in range(len(hid_chs))
        ])
        # 最后一层 hidden → SST
        self.conv_last = nn.Conv2d(hid_chs[0], 1, kernel_size=1)

    def forward(self, x):
        """
        x: [B, T_IN, C, H, W]
        returns preds: [B, T_OUT, 1, H, W]
        """
        B, T, C, H, W = x.shape

        # —— Encoder ——
        enc_states = []
        out = x
        for cell in self.enc_cells:
            h = torch.zeros(B, cell.hid_ch, H, W, device=x.device)
            c = torch.zeros_like(h)
            seq_h = []
            for t in range(T):
                h, c = cell(out[:,t], h, c)
                seq_h.append(h)
            out = torch.stack(seq_h, dim=1)  # [B, T, hid_ch, H, W]
            enc_states.append((h, c))

        # —— Decoder ——
        dec_input  = torch.zeros(B, 1, H, W, device=x.device)  # [B,1,H,W]
        dec_states = enc_states[::-1]  # 反向

        preds = []
        for _ in range(T_OUT):
            h_list, c_list = [], []
            inp = dec_input
            for i, cell in enumerate(self.dec_cells):
                prev_h, prev_c = dec_states[i]
                h, c = cell(inp, prev_h, prev_c)
                inp = h
                h_list.append(h); c_list.append(c)
            dec_states = list(zip(h_list, c_list))
            out_frame  = self.conv_last(h_list[-1])  # [B,1,H,W]
            preds.append(out_frame.unsqueeze(1))     # [B,1,1,H,W]
            dec_input = out_frame

        preds = torch.cat(preds, dim=1)  # [B, T_OUT, 1, H, W]
        return preds


# -------- 4. 自定义 Dataset --------
class SSTDataset(Dataset):
    def __init__(self, ds, ocean_mask):
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 转成 [T, C, H, W]
        x = torch.from_numpy(self.X[idx]).permute(0,3,1,2).float()
        y = torch.from_numpy(self.Y[idx]).permute(0,3,1,2).float()
        # 掩膜转成 [H, W]
        mask = torch.from_numpy(self.mask.values).float()
        return x, y, mask


# -------- 5. 物理损失函数 --------
def gradient_loss(pred, true, mask2):
    """
    pred,true: [B, T_OUT,1,H,W]
    mask2:     [B, H, W]
    """
    # 去掉 channel 维 → [B,T,H,W]
    p = pred.squeeze(2)
    t = true.squeeze(2)
    # 空间梯度
    dx_p = p[:,:,: ,1:] - p[:,:,:, :-1]
    dx_t = t[:,:,: ,1:] - t[:,:,:, :-1]
    dy_p = p[:,:,1:,:] - p[:,:, :-1,:]
    dy_t = t[:,:,1:,:] - t[:,:, :-1,:]
    # 对应掩膜
    mx = mask2[:,:,1:]   # [B, H, W-1]
    my = mask2[:,1: ,:]  # [B, H-1, W]
    # 扩展 time 维后计算
    loss_x = ((dx_p-dx_t)**2 * mx.unsqueeze(1)).mean()
    loss_y = ((dy_p-dy_t)**2 * my.unsqueeze(1)).mean()
    return (loss_x + loss_y)/2

def laplacian_loss(pred, true, mask2):
    """
    pred,true: [B, T_OUT,1,H,W]
    mask2:     [B, H, W]
    """
    B,T,_,H,W = pred.shape
    kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                          dtype=torch.float32, device=pred.device)\
                  .view(1,1,3,3)
    # 合并 B*T 做卷积
    p = pred.view(-1,1,H,W)
    t = true.view(-1,1,H,W)
    lap_p = F.conv2d(p, kernel, padding=1).view(B,T,H,W)
    lap_t = F.conv2d(t, kernel, padding=1).view(B,T,H,W)
    # 掩膜扩 time 维
    m = mask2.unsqueeze(1)  # [B,1,H,W] → broadcast to [B,T,H,W]
    return ((lap_p-lap_t)**2 * m).mean()


# -------- 6. 训练 & 验证 --------
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for x,y,mask in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mask2 = mask.to(DEVICE)  # [B,H,W]
        # 预测
        pred = model(x)           # [B,T_OUT,1,H,W]
        # 基础 MSE（掩膜后）
        mse = F.mse_loss(pred * mask2.unsqueeze(1).unsqueeze(2),
                         y    * mask2.unsqueeze(1).unsqueeze(2))
        # 物理约束
        g_l = gradient_loss(pred, y, mask2)
        v_l = laplacian_loss(pred, y, mask2)
        loss = mse + λ_G*g_l + λ_V*v_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues, all_masks = [], [], []
    for x,y,mask in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mask2 = mask.to(DEVICE)
        pred = model(x)
        mse = F.mse_loss(pred * mask2.unsqueeze(1).unsqueeze(2),
                         y    * mask2.unsqueeze(1).unsqueeze(2))
        g_l  = gradient_loss(pred, y, mask2)
        v_l  = laplacian_loss(pred, y, mask2)
        loss = mse + λ_G*g_l + λ_V*v_l
        total_loss += loss.item()*x.size(0)

        all_preds.append(pred.cpu().numpy())
        all_trues.append(y.cpu().numpy())
        all_masks.append(mask2.cpu().numpy())

    # 全区 RMSE
    preds = np.concatenate(all_preds, axis=0)  # [N,T,1,H,W]
    trues = np.concatenate(all_trues, axis=0)
    masks = np.concatenate(all_masks,axis=0)   # [N,H,W]
    err2  = ((preds.squeeze(2)-trues.squeeze(2))**2 * masks[:,None,:,:]).sum()
    rmse  = np.sqrt(err2 / masks.sum() / preds.shape[1])
    return total_loss / len(loader.dataset), rmse


def main():
    # 1）加载数据
    ds         = xr.open_dataset(DATA_PATH)
    land_mask  = ds['analysed_sst'].isnull().all(dim='time')
    ocean_mask = ~land_mask
    dataset    = SSTDataset(ds, ocean_mask)

    # 2）划分数据集
    N       = len(dataset)
    n_train = int(N*0.7)
    n_val   = int(N*0.15)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2)

    # 3）模型、优化器、调度
    model     = SSTConvLSTM(in_ch=6).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # TensorBoard writer
    writer = SummaryWriter(LOG_DIR)

    # 为可视化存储每 epoch 的指标
    train_losses = []
    val_losses   = []
    val_rmses    = []
    # 4）训练 & 验证
    best_rmse, wait = float('inf'), 0
    for epoch in range(1, EPOCHS+1):
        trn_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_rmse = eval_epoch(model, val_loader)
        print(f"[{epoch:03d}] train_loss={trn_loss:.4f}  val_loss={val_loss:.4f}  val_RMSE={val_rmse:.4f}")
        # —— 记录到 TensorBoard ——
        writer.add_scalar("Loss/train", trn_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss, epoch)
        writer.add_scalar("RMSE/val",   val_rmse, epoch)
        # —— 存入列表以便最后可视化 ——
        train_losses.append(trn_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)


        scheduler.step()
        if val_rmse < best_rmse:
            best_rmse, wait = val_rmse, 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best.pth"))
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # 5）测试集评估
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best.pth")))
    test_loss, test_rmse = eval_epoch(model, test_loader)
    print(f"Test RMSE = {test_rmse:.4f}")



    writer.close()
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("loss_curve.png")   # 保存图片
    plt.close()

    plt.figure()
    plt.plot(epochs, val_rmses, label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Validation RMSE")
    plt.savefig("rmse_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
