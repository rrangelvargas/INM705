import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLanguageAttentionModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1, dropout=0.0, attention_type='linear'):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_type = attention_type.lower()

        # define a bidirectional LSTM to encode the input sequence
        self.lstm = nn.LSTM(
            input_size=225,           # each frame is a 225D vector (flattened landmarks)
            hidden_size=hidden_size,  # LSTM hidden state size
            num_layers=num_layers,    # number of LSTM layers
            batch_first=True,         # input/output tensors have shape (batch, seq, feature)
            bidirectional=True,       # process the sequence forwards and backwards
            dropout=dropout if num_layers > 1 else 0.0  # dropout only if more than 1 layer
        )

        # define the attention mechanism
        if self.attention_type == 'linear':
            # simple linear attention: project LSTM output to a scalar score
            self.attn = nn.Linear(hidden_size * 2, 1)  # [batch, seq_len, 512] → [batch, seq_len, 1]
        elif self.attention_type == 'additive':
            # additive attention: apply tanh before scoring
            self.attn_W = nn.Linear(hidden_size * 2, hidden_size)  # first projection
            self.attn_v = nn.Linear(hidden_size, 1)                # second projection to scalar
        else:
            raise ValueError(f"Unknown attention type: {attention_type}. Use 'linear' or 'additive'.")

        # fully connected layer for final classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # flatten input if 4D (batch, seq_len, 15, 15) → (batch, seq_len, 225)
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)

        # pass input through the LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch, seq_len, hidden_size * 2]

        # apply attention mechanism
        if self.attention_type == 'linear':
            # linear attention: score each timestep directly
            attn_scores = self.attn(lstm_out).squeeze(-1)  # [batch, seq_len]
        elif self.attention_type == 'additive':
            # additive attention: apply tanh activation then score
            energy = torch.tanh(self.attn_W(lstm_out))     # [batch, seq_len, hidden_size]
            attn_scores = self.attn_v(energy).squeeze(-1)  # [batch, seq_len]

        # normalize attention scores across the sequence
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len]

        # compute the weighted sum of LSTM outputs using attention weights
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden_size * 2]

        # predict class label from the context vector
        return self.fc(context)  # [batch, num_classes]
