import os
# 屏蔽 TensorFlow 的 INFO 级别日志（仅 ERROR 会显示）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 关闭 oneDNN 优化（如果你确实想禁用它）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import xarray as xr
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs import *
from data.sst_dataset import SSTDataset
from models.sst_conv_lstm import SSTConvLSTM
from losses.physical_losses import gradient_loss, laplacian_loss
from utils.logger import get_logger

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for x, y, mask in loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        pred = model(x)  # [B, T_OUT, 1, H, W]
        # masked MSE
        mse = torch.nn.functional.mse_loss(
            pred * mask.unsqueeze(1).unsqueeze(2),
            y    * mask.unsqueeze(1).unsqueeze(2)
        )
        # physical losses
        g_l = gradient_loss(pred, y, mask)
        v_l = laplacian_loss(pred, y, mask)
        loss = mse + λ_G * g_l + λ_V * v_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues, all_masks = [], [], []
    for x, y, mask in loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        pred = model(x)
        mse = torch.nn.functional.mse_loss(
            pred * mask.unsqueeze(1).unsqueeze(2),
            y    * mask.unsqueeze(1).unsqueeze(2)
        )
        g_l = gradient_loss(pred, y, mask)
        v_l = laplacian_loss(pred, y, mask)
        loss = mse + λ_G * g_l + λ_V * v_l
        total_loss += loss.item() * x.size(0)

        all_preds.append(pred.cpu())
        all_trues.append(y.cpu())
        all_masks.append(mask.cpu())

    # compute overall RMSE
    preds = torch.cat(all_preds, dim=0).squeeze(2).numpy()    # [N, T_OUT, H, W]
    trues = torch.cat(all_trues, dim=0).squeeze(2).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()              # [N, H, W]
    err2  = ((preds - trues)**2 * masks[:,None,:,:]).sum()
    rmse  = (err2 / (masks.sum() * preds.shape[1]))**0.5

    return total_loss / len(loader.dataset), rmse

def main():
    # 1. load dataset
    ds = xr.open_dataset(DATA_PATH)
    ocean_mask = ~ds['analysed_sst'].isnull().all(dim='time')
    dataset = SSTDataset(ds, ocean_mask, T_IN, T_OUT)

    # 2. split & dataloaders
    N = len(dataset)
    n_train = int(N * 0.7)
    n_val   = int(N * 0.15)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. model, optimizer, scheduler, logger
    model     = SSTConvLSTM(in_ch=6, hid_chs=[64,128], T_OUT=T_OUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    writer = get_logger(LOG_DIR)

    # lists to store epoch metrics
    train_losses, val_losses, val_rmses = [], [], []

    best_rmse, wait = float('inf'), 0
    for epoch in range(1, EPOCHS+1):
        trn_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_rmse = eval_epoch(model, val_loader)

        # console
        print(f"[{epoch:02d}] train_loss={trn_loss:.4f}  val_loss={val_loss:.4f}  val_RMSE={val_rmse:.4f}")

        # TensorBoard
        writer.add_scalar("Loss/train", trn_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss, epoch)
        writer.add_scalar("RMSE/val",   val_rmse, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # record for offline plot
        train_losses.append(trn_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        # adjust LR & checkpointing
        scheduler.step(val_rmse)
        if val_rmse < best_rmse:
            best_rmse, wait = val_rmse, 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best.pth"))
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    writer.close()

    # 4. test evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best.pth")))
    _, test_rmse = eval_epoch(model, test_loader)
    print(f"Test RMSE = {test_rmse:.4f}")

    # 5. offline visualization - save curves
    epochs = list(range(1, len(train_losses)+1))
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("loss_curve.png"); plt.close()

    plt.figure()
    plt.plot(epochs, val_rmses, label="Val RMSE")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.legend()
    plt.title("Validation RMSE")
    plt.savefig("rmse_curve.png"); plt.close()

if __name__=="__main__":
    main()
