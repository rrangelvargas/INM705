import torch
import torch.nn as nn
import math

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

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            lr = [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
        
        for param_group, lr in zip(self.optimizer.param_groups, lr):
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups] 