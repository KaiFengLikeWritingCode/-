import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SeaIceDataset(Dataset):
    def __init__(self, img_paths, lbl_paths, transform=None):
        self.imgs = img_paths
        self.lbls = lbl_paths
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.imgs[idx]), dtype=np.uint8)
        lbl = np.array(Image.open(self.lbls[idx]), dtype=np.uint8)
        # 灰度→3通道
        img = np.stack([img, img, img], axis=-1)

        if self.transform:
            aug = self.transform(image=img, mask=lbl)
            img, lbl = aug["image"].float() / 255.0, aug["mask"].long()
        else:
            img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            lbl = torch.from_numpy(lbl).long()
        return img, lbl
