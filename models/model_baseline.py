import torch
import torch.nn as nn

# simple BiLSTM model for sign language word classification
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size  # store hidden size for use later

        # define a bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=225,           # flattened landmarks per frame
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,         # input format: (batch, seq_len, feature_dim)
            bidirectional=True,       # use both forward and backward context
            dropout=dropout if num_layers > 1 else 0.0  # only apply dropout if more than 1 layer
        )

        # fully connected layer to project LSTM output to number of classes
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)  # bidirectional -> hidden_size * 2

    def forward(self, x):
        # flatten input if extra dimensions present
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)

        # pass input through LSTM
        x, _ = self.lstm(x)

        # classify based on the last timestep's output
        return self.fc(x[:, -1])
