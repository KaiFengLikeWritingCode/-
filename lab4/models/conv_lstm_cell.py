import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hid_ch = hid_ch
        self.conv   = nn.Conv2d(in_ch + hid_ch, 4*hid_ch, kernel_size,
                                padding=padding, bias=True)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        i,f,o,g = torch.split(self.conv(combined), self.hid_ch, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f*c + i*g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
