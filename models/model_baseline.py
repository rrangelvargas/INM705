import torch.nn as nn  # import neural network modules from torch

class SignLanguageModel(nn.Module):  # define model class
    def __init__(self, num_classes):  # init model
        super().__init__()  # call super
        self.lstm = nn.LSTM(  # define lstm
            input_size=225,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(512, num_classes)  # define final layer

    def forward(self, x):  # define forward pass
        if x.dim() == 4:  # flatten if needed
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)  # pass through lstm
        return self.fc(x[:, -1])  # return last step output
