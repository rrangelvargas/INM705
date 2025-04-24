import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + (self.smoothing / n_classes)
        log_prob = F.log_softmax(pred, dim=1)
        loss = (-smooth_one_hot * log_prob).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class AdditiveAngularMarginLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace) for better feature learning."""
    def __init__(self, margin=0.5, scale=30.0, reduction='mean'):
        super(AdditiveAngularMarginLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, target):
        # Normalize the predictions
        pred_norm = F.normalize(pred, p=2, dim=1)
        
        # Get the cosine of the angles
        cosine = pred_norm
        
        # Calculate the sine of the angles
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Calculate the cosine of the angle plus margin
        phi = cosine * torch.cos(torch.tensor(self.margin, device=cosine.device)) - sine * torch.sin(torch.tensor(self.margin, device=cosine.device))
        
        # Create one-hot encoding of targets
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Calculate the output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Calculate the loss
        loss = F.cross_entropy(output, target, reduction=self.reduction)
        
        return loss


def get_loss(loss_type, **kwargs):
    """Factory function to get the appropriate loss function."""
    if loss_type == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == "arcface":
        return AdditiveAngularMarginLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 