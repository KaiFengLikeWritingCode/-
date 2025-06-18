import torch
import torch.nn.functional as F
from torch import Tensor


def gradient_loss(pred: Tensor,
                  true: Tensor,
                  mask2: Tensor) -> Tensor:
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

def laplacian_loss(pred: Tensor,
                   true: Tensor,
                   mask2: Tensor) -> Tensor:
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
