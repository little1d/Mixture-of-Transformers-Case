"""
Image and Text Encoders for Multi-modal Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TextEncoder(nn.Module):
    """Simple text encoder based on embedding + transformer"""

    def __init__(
        self,
        vocab_size: int = 256,  # Character-level vocabulary
        hidden_dim: int = 512,
        max_length: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: [batch_size, seq_length]
        Returns:
            text_features: [batch_size, hidden_dim]
        """
        batch_size, seq_length = text_tokens.shape

        # Create position indices
        positions = torch.arange(seq_length, device=text_tokens.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_emb = self.token_embedding(text_tokens)
        pos_emb = self.position_embedding(positions)
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)

        # Attention mask (ignore padding tokens)
        attention_mask = text_tokens == 0  # 0 is padding token

        # Transformer encoding
        encoded = self.transformer(embeddings, src_key_padding_mask=attention_mask)

        # Global pooling (mean of non-padding tokens)
        mask = (~attention_mask).float().unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)

        # Output projection
        text_features = self.output_projection(pooled)

        return text_features


class ImageEncoder(nn.Module):
    """Vision Transformer-based image encoder"""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Calculate number of patches
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, hidden_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, channels, height, width]
        Returns:
            image_features: [batch_size, hidden_dim]
        """
        batch_size = images.shape[0]

        # Extract patches
        patches = self._extract_patches(images)  # [batch_size, n_patches, patch_dim]

        # Patch embeddings
        patch_embeddings = self.patch_embedding(
            patches
        )  # [batch_size, n_patches, hidden_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embedding
        embeddings = self.dropout(embeddings)

        # Transformer encoding
        encoded = self.transformer(embeddings)

        # Use CLS token as image representation
        cls_output = encoded[:, 0]

        # Output projection
        image_features = self.output_projection(cls_output)

        return image_features

    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patches from images"""
        batch_size, channels, height, width = images.shape

        # Reshape to patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(
            batch_size, channels, -1, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(
            batch_size, -1, channels * self.patch_size * self.patch_size
        )

        return patches


class MultiModalEncoder(nn.Module):
    """Combines text and image encoders"""

    def __init__(self, config):
        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=256,  # Character-level
            hidden_dim=config.hidden_dim,
            max_length=config.text_max_length,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        self.image_encoder = ImageEncoder(
            image_size=config.image_size,
            patch_size=config.image_patch_size,
            in_channels=config.image_channels,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

    def forward(
        self, images: torch.Tensor, text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images and text separately

        Args:
            images: [batch_size, channels, height, width]
            text_tokens: [batch_size, seq_length]

        Returns:
            image_features: [batch_size, hidden_dim]
            text_features: [batch_size, hidden_dim]
        """
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text_tokens)

        return image_features, text_features
