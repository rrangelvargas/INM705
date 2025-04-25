import torch
import torch.nn as nn
import torch.nn.functional as F

# simple BiLSTM model for sequence classification
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size  # store hidden size for internal use

        # define a bidirectional LSTM to encode input sequences
        self.lstm = nn.LSTM(
            input_size=225,           # input dimension (flattened landmarks)
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,         # (batch, seq_len, feature_dim)
            bidirectional=True,       # concatenate forward and backward outputs
            dropout=dropout if num_layers > 1 else 0.0  # dropout only if multiple layers
        )

        # fully connected layer maps LSTM output to class logits
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        # flatten input if extra dimensions exist
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)

        # pass through LSTM
        x, _ = self.lstm(x)

        # use output from the last timestep for classification
        return self.fc(x[:, -1])


# label smoothing loss for regularization
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing  # smoothing factor
        self.reduction = reduction  # reduction type: mean or sum

    def forward(self, pred, target):
        n_classes = pred.size(1)  # number of output classes

        # create one-hot encoded target
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # apply label smoothing
        smooth_one_hot = one_hot * (1 - self.smoothing) + (self.smoothing / n_classes)

        # compute log probabilities
        log_prob = F.log_softmax(pred, dim=1)

        # calculate smoothed cross-entropy loss
        loss = (-smooth_one_hot * log_prob).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# additive angular margin loss (ArcFace) for better feature separation
class AdditiveAngularMarginLoss(nn.Module):
    def __init__(self, margin=0.35, scale=64.0, reduction='mean'):
        super(AdditiveAngularMarginLoss, self).__init__()
        self.margin = margin  # margin added to angles
        self.scale = scale    # scale factor applied to logits
        self.reduction = reduction

    def forward(self, pred, target):
        # normalize predictions to unit vectors
        pred_norm = F.normalize(pred, p=2, dim=1)

        # cosine similarity between features and class centers
        cosine = pred_norm

        # calculate sine from cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # compute margin-augmented cosine value
        phi = cosine * torch.cos(torch.tensor(self.margin, device=cosine.device)) - sine * torch.sin(torch.tensor(self.margin, device=cosine.device))

        # create one-hot encoding for targets
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(1, target.unsqueeze(1), 1)

        # mix phi and cosine depending on whether the class matches
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale  # scale up

        # compute cross-entropy loss on modified logits
        loss = F.cross_entropy(output, target, reduction=self.reduction)

        return loss


# combined loss: label smoothing + ArcFace
class CombinedLoss(nn.Module):
    def __init__(self, smoothing=0.1, margin=0.5, scale=30.0, reduction='mean', alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.label_smoothing = LabelSmoothingLoss(smoothing=smoothing, reduction=reduction)
        self.arcface = AdditiveAngularMarginLoss(margin=margin, scale=scale, reduction=reduction)
        self.alpha = alpha  # weight between the two losses

    def forward(self, pred, target):
        # compute individual losses
        ls_loss = self.label_smoothing(pred, target)
        arc_loss = self.arcface(pred, target)

        # weighted combination of losses
        return self.alpha * ls_loss + (1 - self.alpha) * arc_loss


# utility function to select loss function based on string
def get_loss(loss_type, **kwargs):
    if loss_type == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == "arcface":
        return AdditiveAngularMarginLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
