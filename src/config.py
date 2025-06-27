"""
Configuration for MoT Image-Text Matching Experiment
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration"""

    # Dataset paths
    data_root: str = "data"
    train_images_dir: str = "train/images"
    val_images_dir: str = "val/images"
    train_captions_file: str = "train/captions.json"
    val_captions_file: str = "val/captions.json"

    # Data processing
    max_text_length: int = 128
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    # Negative sampling
    negative_ratio: float = 1.0  # 1:1 positive:negative ratio


@dataclass
class ModelConfig:
    """Model configuration"""

    # Shared parameters
    hidden_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    head_dim: int = 64
    dropout: float = 0.1

    # Text encoder
    text_vocab_size: int = 30522  # BERT vocab size
    text_max_length: int = 128

    # Image encoder
    image_patch_size: int = 16
    image_size: int = 224
    image_channels: int = 3

    # MoT specific
    n_modalities: int = 2  # text + image

    # Classification
    num_classes: int = 2  # match/no-match


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_epochs: int = 20

    # Regularization
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1

    # Checkpointing
    save_every: int = 5
    eval_every: int = 1

    # Device
    device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    # Experiment meta
    experiment_name: str = "mot_vs_transformer"
    output_dir: str = "experiments"
    seed: int = 42

    # Comparison settings
    compare_models: bool = True
    measure_flops: bool = True
    measure_inference_time: bool = True
