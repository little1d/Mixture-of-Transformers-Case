"""
MoT and Traditional Transformer Models for Multi-modal Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .encoders import MultiModalEncoder


class ModalityUntiedFeedForward(nn.Module):
    """
    Modality-specific feed-forward networks for MoT architecture
    Based on the official MoT implementation
    """

    def __init__(
        self, dim: int, hidden_dim: int, dropout: float = 0.1, n_modalities: int = 2
    ):
        super().__init__()
        self.n_modalities = n_modalities

        # Create modality-specific feed-forward networks
        self.local_experts_w1 = nn.ModuleList(
            [nn.Linear(dim, hidden_dim, bias=False) for _ in range(n_modalities)]
        )
        self.local_experts_w2 = nn.ModuleList(
            [nn.Linear(hidden_dim, dim, bias=False) for _ in range(n_modalities)]
        )
        self.local_experts_w3 = nn.ModuleList(
            [nn.Linear(dim, hidden_dim, bias=False) for _ in range(n_modalities)]
        )

        # Modality-specific normalization
        self.local_experts_ffn_norm = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(n_modalities)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, modality_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size * seq_len, dim] - flattened multimodal tokens
            modality_masks: list of boolean masks for each modality
        """
        expert_outputs = []

        for i in range(self.n_modalities):
            if modality_masks[i].sum() > 0:  # Check if modality has tokens
                expert_input = x[modality_masks[i]]

                # SwiGLU activation: w1(x) * swish(w3(x)) -> w2
                w1_out = self.local_experts_w1[i](expert_input)
                w3_out = self.local_experts_w3[i](expert_input)
                hidden = w1_out * F.silu(w3_out)  # SwiGLU
                hidden = self.dropout(hidden)

                expert_output = self.local_experts_w2[i](hidden)
                expert_output = self.local_experts_ffn_norm[i](expert_output)
                expert_outputs.append((expert_output, modality_masks[i]))

        # Merge outputs back
        output = torch.zeros_like(x)
        for expert_output, mask in expert_outputs:
            output[mask] = expert_output

        return output


class ModalityUntiedAttention(nn.Module):
    """
    Modality-specific attention for MoT architecture
    Based on the official MoT implementation
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        n_modalities: int = 2,
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        # Modality-specific projections
        self.local_experts_wq = nn.ModuleList(
            [
                nn.Linear(dim, n_heads * head_dim, bias=False)
                for _ in range(n_modalities)
            ]
        )
        self.local_experts_wk = nn.ModuleList(
            [
                nn.Linear(dim, n_heads * head_dim, bias=False)
                for _ in range(n_modalities)
            ]
        )
        self.local_experts_wv = nn.ModuleList(
            [
                nn.Linear(dim, n_heads * head_dim, bias=False)
                for _ in range(n_modalities)
            ]
        )
        self.local_experts_wo = nn.ModuleList(
            [
                nn.Linear(n_heads * head_dim, dim, bias=False)
                for _ in range(n_modalities)
            ]
        )

        # Modality-specific normalization
        self.local_experts_attention_norm = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(n_modalities)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_masks: List[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size * seq_len, dim]
            modality_masks: list of boolean masks for each modality
            attn_mask: attention mask
        """
        batch_seq_len, dim = x.shape

        # Process Q, K, V for each modality
        all_q, all_k, all_v = [], [], []
        modality_indices = []

        for i in range(self.n_modalities):
            if modality_masks[i].sum() > 0:
                expert_input = x[modality_masks[i]]

                q = self.local_experts_wq[i](expert_input)
                k = self.local_experts_wk[i](expert_input)
                v = self.local_experts_wv[i](expert_input)

                all_q.append(q)
                all_k.append(k)
                all_v.append(v)
                modality_indices.append((i, modality_masks[i]))

        # Concatenate all Q, K, V
        if len(all_q) > 0:
            merged_q = torch.cat(all_q, dim=0)
            merged_k = torch.cat(all_k, dim=0)
            merged_v = torch.cat(all_v, dim=0)

            # Reshape for multi-head attention
            def reshape_for_attention(tensor):
                batch_len = tensor.shape[0]
                return tensor.view(batch_len, self.n_heads, self.head_dim).transpose(
                    0, 1
                )

            q = reshape_for_attention(merged_q)  # [n_heads, total_tokens, head_dim]
            k = reshape_for_attention(merged_k)
            v = reshape_for_attention(merged_v)

            # Compute attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            attn_output = torch.matmul(
                attn_weights, v
            )  # [n_heads, total_tokens, head_dim]
            attn_output = attn_output.transpose(
                0, 1
            ).contiguous()  # [total_tokens, n_heads, head_dim]
            attn_output = attn_output.view(
                -1, self.n_heads * self.head_dim
            )  # [total_tokens, dim]

            # Split back to modalities and apply output projections
            output = torch.zeros_like(x)
            start_idx = 0

            for modality_idx, mask in modality_indices:
                end_idx = start_idx + mask.sum().item()
                modality_output = attn_output[start_idx:end_idx]

                # Apply modality-specific output projection and normalization
                modality_output = self.local_experts_wo[modality_idx](modality_output)
                modality_output = self.local_experts_attention_norm[modality_idx](
                    modality_output
                )

                output[mask] = modality_output
                start_idx = end_idx
        else:
            output = x

        return output


class MoTTransformerBlock(nn.Module):
    """Single MoT transformer block"""

    def __init__(self, config):
        super().__init__()

        self.attention = ModalityUntiedAttention(
            dim=config.hidden_dim,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
            n_modalities=config.n_modalities,
        )

        self.feed_forward = ModalityUntiedFeedForward(
            dim=config.hidden_dim,
            hidden_dim=config.hidden_dim * 4,
            dropout=config.dropout,
            n_modalities=config.n_modalities,
        )

    def forward(
        self, x: torch.Tensor, modality_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        # Residual connection around attention
        x = x + self.attention(x, modality_masks)

        # Residual connection around feed-forward
        x = x + self.feed_forward(x, modality_masks)

        return x


class TraditionalTransformerBlock(nn.Module):
    """Traditional transformer block for comparison"""

    def __init__(self, config):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(config.hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

        self.ffn_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for attention [batch_size * seq_len, dim] -> [batch_size, seq_len, dim]
        # For simplicity, assume batch_size = 1 and treat as [1, seq_len, dim]
        seq_len = x.shape[0]
        x = x.unsqueeze(0)  # [1, seq_len, dim]

        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.attention_norm(x)

        # Feed-forward with residual connection
        ffn_output = self.feed_forward(x)
        x = x + ffn_output
        x = self.ffn_norm(x)

        # Reshape back
        x = x.squeeze(0)  # [seq_len, dim]

        return x


class MoTModel(nn.Module):
    """Complete MoT model for image-text matching"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoders
        self.encoder = MultiModalEncoder(config)

        # MoT transformer layers
        self.transformer_layers = nn.ModuleList(
            [MoTTransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]

        # Encode images and text
        image_features, text_features = self.encoder(images, text_tokens)

        # Stack features and create modality masks
        multimodal_features = torch.stack(
            [image_features, text_features], dim=1
        )  # [batch, 2, dim]
        multimodal_features = multimodal_features.view(
            batch_size * 2, -1
        )  # [batch*2, dim]

        # Create modality masks
        modality_masks = [
            torch.tensor(
                [True, False] * batch_size, device=images.device
            ),  # Image mask
            torch.tensor([False, True] * batch_size, device=images.device),  # Text mask
        ]

        # Pass through MoT transformer layers
        for layer in self.transformer_layers:
            multimodal_features = layer(multimodal_features, modality_masks)

        # Reshape back and combine features
        multimodal_features = multimodal_features.view(batch_size, 2, -1)
        combined_features = multimodal_features.view(batch_size, -1)  # [batch, dim*2]

        # Classification
        logits = self.classifier(combined_features)

        return logits


class TraditionalTransformerModel(nn.Module):
    """Traditional transformer model for comparison"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoders
        self.encoder = MultiModalEncoder(config)

        # Traditional transformer layers
        self.transformer_layers = nn.ModuleList(
            [TraditionalTransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]

        # Encode images and text
        image_features, text_features = self.encoder(images, text_tokens)

        # Stack features
        multimodal_features = torch.stack(
            [image_features, text_features], dim=1
        )  # [batch, 2, dim]
        multimodal_features = multimodal_features.view(
            batch_size * 2, -1
        )  # [batch*2, dim]

        # Pass through traditional transformer layers
        for layer in self.transformer_layers:
            multimodal_features = layer(multimodal_features)

        # Reshape back and combine features
        multimodal_features = multimodal_features.view(batch_size, 2, -1)
        combined_features = multimodal_features.view(batch_size, -1)  # [batch, dim*2]

        # Classification
        logits = self.classifier(combined_features)

        return logits
