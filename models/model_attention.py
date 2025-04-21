import torch
import torch.nn as nn

class SignLanguageAttentionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=225,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)
            
        # Get LSTM outputs for all timesteps
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights to get context vector
        context = torch.bmm(attention_weights.transpose(1, 2), lstm_out)  # [batch, 1, hidden*2]
        context = context.squeeze(1)  # [batch, hidden*2]
        
        # Final classification
        return self.fc(context)