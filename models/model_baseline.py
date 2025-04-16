import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=225,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.layernorm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        pooled = x.mean(dim=1)  # or x[:, -1]
        return self.fc(pooled)
