# src/evaluate.py
"""
Batch inference  ——  轮廓线 + 标签
---------------------------------
python src/evaluate.py \
       --config configs/default.yaml \
       --img_dir data/test_ssha_png
"""
import argparse, yaml
from pathlib import Path

import numpy as np
import torch, cv2
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from model import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Contour-line overlay for sea-ice segmentation")
    p.add_argument("--config",  type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--img_dir", type=Path, required=True,
                   help="Directory containing grayscale PNG images")
    return p.parse_args()


# 兼容 DeepLab(dict) vs Unet(tensor)
get_logits = lambda o: (o["out"] if isinstance(o, dict) else o)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    model_cfg = cfg["model"]
    eval_cfg  = cfg["evaluation"]

    device = eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    result_dir = Path(eval_cfg.get("result_dir", "results_contour"))
    result_dir.mkdir(exist_ok=True, parents=True)

    # ---------- 载入模型 ----------
    model = get_model(
        name                = model_cfg["name"],
        num_classes         = model_cfg["num_classes"],
        backbone            = model_cfg.get("backbone", "resnet34"),
        pretrained_backbone = model_cfg.get("pretrained_backbone", False)
    )
    model.load_state_dict(torch.load(eval_cfg["weights"], map_location=device))
    model.to(device).eval()

    # 颜色与文本样式 (BGR)
    color_map = {1: (255, 0, 0),   # Ice  - blue line
                 2: (0, 255, 0)}   # Land - green line
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fsize = 0.5; fthick = 1

    # ---------- 批量推理 ----------
    for img_path in tqdm(sorted(args.img_dir.glob("*.png"))):
        gray = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        inp  = ToTensor()(np.stack([gray]*3, -1)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = get_logits(model(inp)).softmax(1).argmax(1).cpu().numpy()[0]

        # 基于原图创建着色层
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 仅处理 label 1, 2
        for lbl in (1, 2):
            mask = (pred == lbl).astype(np.uint8)
            if mask.sum() == 0:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color_map[lbl], 1)

            # 给每个轮廓放置标签文本
            for cnt in contours:
                m = cv2.moments(cnt)
                if m["m00"] == 0:           # 避免除 0
                    continue
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                label_text = "Ice" if lbl == 1 else "Land"
                cv2.putText(canvas, label_text, (cx, cy),
                            font, fsize, color_map[lbl], fthick, cv2.LINE_AA)

        cv2.imwrite(str(result_dir / f"overlay_{img_path.stem}.png"), canvas)


if __name__ == "__main__":
    main()
