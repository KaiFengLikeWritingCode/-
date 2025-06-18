import torch
import torch.nn as nn
from .conv_lstm_cell import ConvLSTMCell

class SSTConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_chs=[64,128], T_OUT=7):
        super().__init__()
        self.hid_chs = hid_chs
        self.T_OUT   = T_OUT
        # build encoder & decoder
        self.enc_cells = nn.ModuleList([ConvLSTMCell(in_ch if i==0 else hid_chs[i-1], hid_chs[i])
                                        for i in range(len(hid_chs))])
        self.dec_cells = nn.ModuleList([ConvLSTMCell(1 if i==0 else hid_chs[-i], hid_chs[-i-1])
                                        for i in range(len(hid_chs))])
        self.conv_last = nn.Conv2d(hid_chs[0], 1, kernel_size=1)

    def forward(self, x):
        B,T,C,H,W = x.shape
        enc_states, out = [], x
        # encoder loop
        for cell in self.enc_cells:
            h = torch.zeros(B, cell.hid_ch, H, W, device=x.device)
            c = torch.zeros_like(h)
            seq_h = []
            for t in range(T):
                h, c = cell(out[:,t], h, c)
                seq_h.append(h)
            out = torch.stack(seq_h, dim=1)
            enc_states.append((h,c))
        # decoder loop
        dec_input  = torch.zeros(B,1,H,W, device=x.device)
        dec_states = enc_states[::-1]
        preds = []
        for _ in range(self.T_OUT):
            h_list, c_list = [], []
            inp = dec_input
            for i, cell in enumerate(self.dec_cells):
                prev_h, prev_c = dec_states[i]
                h, c = cell(inp, prev_h, prev_c)
                inp = h; h_list.append(h); c_list.append(c)
            dec_states = list(zip(h_list,c_list))
            out_frame  = self.conv_last(h_list[-1])
            preds.append(out_frame.unsqueeze(1))
            dec_input = out_frame
        return torch.cat(preds, dim=1)
