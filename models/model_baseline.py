import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size  # store for use later

        self.lstm = nn.LSTM(
            input_size=225,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(self.hidden_size * 2, num_classes)  # bidirectional = hidden * 2

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)

        x, _ = self.lstm(x)
        return self.fc(x[:, -1])  # use last time step's output
