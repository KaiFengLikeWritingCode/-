"""
Training script for Sea-Ice semantic segmentation.

Usage:
    python src/train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import SeaIceDataset
from model import get_model
from utils import iou_metric


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Train Sea-Ice segmentation")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                   help="Path to YAML config file")
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def freeze_bn_layers(model: torch.nn.Module):
    """Freeze all nn.BatchNorm2d layers (useful for small batch)."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # -------------------------------- data paths & logging ----------------------------
    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["train"], cfg["model"]

    img_dir, lbl_dir = Path(data_cfg["img_dir"]), Path(data_cfg["lbl_dir"])
    out_dir = Path(train_cfg.get("out_dir", "results"))
    tb_dir = out_dir / "tb_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log"),
        ],
    )
    logger = logging.getLogger("train")

    writer = SummaryWriter(tb_dir)

    # -------------------------------- dataset ----------------------------------------
    imgs = sorted(img_dir.glob("*.png"))
    lbls = sorted(lbl_dir.glob("*.png"))
    dataset = SeaIceDataset(imgs, lbls, transform=None)

    trn_n = int(len(dataset) * train_cfg["train_ratio"])
    val_n = len(dataset) - trn_n
    train_ds, val_ds = random_split(
        dataset,
        [trn_n, val_n],
        generator=torch.Generator().manual_seed(train_cfg.get("seed", 42)),
    )

    dl_kwargs = dict(
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        batch_size=train_cfg["batch_size"],
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # -------------------------------- model ------------------------------------------
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        name=model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        backbone=model_cfg.get("backbone", "resnet34"),
        pretrained_backbone=model_cfg.get("pretrained_backbone", False),
    ).to(device)

    if train_cfg.get("freeze_bn", False):
        freeze_bn_layers(model)

    # -------------------------------- optim ------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # -------------------------------- loop -------------------------------------------
    ckpt_intv = train_cfg.get("save_interval_epochs", 1)
    best_miou = 0.0

    for epoch in range(1, train_cfg["epochs"] + 1):
        # -------------------- train --------------------
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)["out"]          # ★ 输出接口已统一
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_ds)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        # -------------------- val ----------------------
        model.eval()
        running_iou = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                x, y = x.to(device), y.to(device)
                logits = model(x)["out"]
                running_iou += (
                    iou_metric(logits, y, model_cfg["num_classes"]).item()
                    * x.size(0)
                )

        epoch_miou = running_iou / len(val_ds)
        writer.add_scalar("mIoU/val", epoch_miou, epoch)

        logger.info(
            f"Epoch {epoch:02d} | train_loss={epoch_loss:.4f} | val_mIoU={epoch_miou:.4f}"
        )

        # checkpoint
        if epoch % ckpt_intv == 0:
            ckpt = out_dir / f"{model_cfg['name']}_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")

        # best model
        if epoch_miou > best_miou:
            best_miou = epoch_miou
            best_file = out_dir / "best_model.pth"
            torch.save(model.state_dict(), best_file)
            logger.info(f"Saved best model: {best_file} (mIoU={best_miou:.4f})")

    writer.close()
    logger.info("Training completed.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
