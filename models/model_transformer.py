import torch
import torch.nn as nn

class TransformerSignLanguageModel(nn.Module):
    def __init__(self, num_classes, seq_len=30, input_dim=225, d_model=512, nhead=8, num_layers=2, dim_feedforward=1024):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)  # project input to model dimension
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))  # learnable positional embeddings

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # use [B, T, C] format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)  # flatten to [B, T, 225]

        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_embedding[:, :x.size(1)]  # add positional encoding

        out = self.transformer_encoder(x)  # [B, T, d_model]
        out = out.mean(dim=1)  # average pooling over time

        return self.classifier(out)  # [B, num_classes]
