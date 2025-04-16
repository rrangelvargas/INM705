import torch
import torch.nn as nn
import torch.nn.functional as F

# VectorQuantizer implements the core of VQ-VAE
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim  # dimensionality of each embedding vector
        self.num_embeddings = num_embeddings  # number of discrete vectors in the codebook
        self.commitment_cost = commitment_cost  # beta term in VQ-VAE loss

        # codebook: learnable embedding matrix of shape [num_embeddings, embedding_dim]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)  # init weights
        print(f"[VQ] Initialized with {num_embeddings} embeddings of dim {embedding_dim}")

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, embedding_dim]
        # flatten to [batch_size * seq_len, embedding_dim] for vector quantization
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # compute squared L2 distance between input vectors and embeddings
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)  # ||z||²
            + torch.sum(self.embedding.weight**2, dim=1)  # ||e||²
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())  # 2 * zᵀe
        )

        # get indices of closest embeddings (codebook vector) for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*T, 1]

        # convert indices to one-hot encodings
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # use encodings to lookup quantized vectors from codebook
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        # compute vector quantization loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # encoder commitment loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # embedding update loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # use straight-through estimator to preserve gradient for encoder
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss  # return quantized vectors and VQ loss

# SignLanguageVQVAEModel wraps the encoder, VQ layer, decoder, and classifier
class SignLanguageVQVAEModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        print("[Model] Initializing VQ-VAE model...")

        # encoder LSTM: encodes input sequences into continuous latent vectors
        self.encoder = nn.LSTM(
            input_size=225,     # number of features per frame
            hidden_size=256,    # LSTM hidden dimension
            num_layers=1,       # single LSTM layer
            batch_first=True,   # input shape is [B, T, C]
            bidirectional=True  # concatenate forward + backward outputs → 512 dim
        )

        # vector quantization layer
        self.vq_layer = VectorQuantizer(
            num_embeddings=128,    # size of codebook
            embedding_dim=512,     # must match encoder output dim
            commitment_cost=0.25   # beta value for VQ loss
        )

        # decoder LSTM: tries to reconstruct input or compress features
        self.decoder = nn.LSTM(
            input_size=512,     # quantized latent vectors
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # classifier: final linear layer for prediction
        self.classifier = nn.Linear(512, num_classes)

        print(f"[Model] VQ-VAE initialized with classifier output dim: {num_classes}")

    def forward(self, x):
        # flatten if input has extra dims (e.g. [B, T, 15, 15])
        if x.dim() == 4:
            batch_size, seq_len, *_ = x.size()
            x = x.view(batch_size, seq_len, -1)  # [B, T, 225]

        # encode sequence
        enc_out, _ = self.encoder(x)  # [B, T, 512]

        # quantize latent vectors
        quantized, vq_loss = self.vq_layer(enc_out)  # [B, T, 512], scalar

        # decode the quantized sequence
        dec_out, _ = self.decoder(quantized)  # [B, T, 512]

        # classify using last time step's output
        logits = self.classifier(dec_out[:, -1])  # [B, num_classes]

        return logits, vq_loss  # prediction and additional VQ loss
