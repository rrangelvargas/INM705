import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLanguageAttentionModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1, dropout=0.0, attention_type='dot'):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type.lower()
        
        self.lstm = nn.LSTM(
            input_size=225,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if self.attention_type == 'dot':
            self.attn = nn.Linear(hidden_size * 2, 1)  # [B, T, 512] -> [B, T, 1]
        elif self.attention_type == 'additive':
            self.attn_W = nn.Linear(hidden_size * 2, hidden_size)
            self.attn_v = nn.Linear(hidden_size, 1)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}. Use 'dot' or 'additive'.")

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(x)  # [B, T, H*2]

        if self.attention_type == 'dot':
            # Dot-style attention
            attn_scores = self.attn(lstm_out).squeeze(-1)  # [B, T]
        elif self.attention_type == 'additive':
            # Additive-style attention
            energy = torch.tanh(self.attn_W(lstm_out))     # [B, T, H]
            attn_scores = self.attn_v(energy).squeeze(-1)  # [B, T]

        attn_weights = F.softmax(attn_scores, dim=1)               # [B, T]
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, H*2]

        return self.fc(context)  # [B, num_classes]
