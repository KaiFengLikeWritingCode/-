import torch.nn as nn

class JointLSTM(nn.Module):
    """
    JointLSTM: 输入通道 in_ch，隐藏层 hidden，层数 layers，
    输出 horizon 步预测，dropout 比例 drop
    """
    def __init__(self, in_ch, hidden, layers, horizon, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            in_ch, hidden, layers,
            batch_first=True,
            dropout=dropout
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last))
