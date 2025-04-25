import torch
import torch.nn as nn
import math

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
            batch_first=True,         # (batch, seq_len, feature_dim)
            bidirectional=True,       # use both forward and backward context
            dropout=dropout if num_layers > 1 else 0.0  # only apply dropout if >1 layer
        )

        # final classifier layer
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)  # bidirectional -> hidden_size * 2

    def forward(self, x):
        # flatten input if extra dimensions present
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)

        # pass through LSTM
        x, _ = self.lstm(x)

        # classify based on the last timestep output
        return self.fc(x[:, -1])


# learning rate scheduler: linear warmup + cosine decay
class WarmupCosineScheduler:
    """Learning rate scheduler with warmup phase and cosine annealing decay."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0):
        self.optimizer = optimizer  # optimizer to adjust
        self.warmup_epochs = warmup_epochs  # number of warmup epochs
        self.max_epochs = max_epochs        # total training epochs
        self.min_lr = min_lr                # minimum learning rate floor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]  # store initial LRs
        self.current_epoch = 0  # initialize epoch counter

    def step(self):
        self.current_epoch += 1  # advance epoch counter
        
        if self.current_epoch < self.warmup_epochs:
            # linear warmup phase
            alpha = self.current_epoch / self.warmup_epochs
            lr = [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # cosine decay phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
        
        # apply new learning rates
        for param_group, lr in zip(self.optimizer.param_groups, lr):
            param_group['lr'] = lr

    def get_last_lr(self):
        # retrieve current learning rates
        return [group['lr'] for group in self.optimizer.param_groups]
