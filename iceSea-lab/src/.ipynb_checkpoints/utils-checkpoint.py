import torch

def iou_metric(pred_logits, labels, n_cls, eps=1e-6):
    preds = pred_logits.argmax(dim=1)
    ious = []
    for c in range(n_cls):
        inter = ((preds==c)&(labels==c)).sum().float()
        uni   = ((preds==c)|(labels==c)).sum().float()
        ious.append(inter/(uni+eps))
    return torch.stack(ious).mean()
