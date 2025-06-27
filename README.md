# Mixture-of-Transformers Case Study

A comprehensive comparison of Mixture-of-Transformers (MoT) vs Traditional Transformer architectures for multimodal image-text matching tasks.

## 🎯 Project Overview

This project implements and compares two transformer architectures:

1. **MoT (Mixture-of-Transformers)**: Uses modality-specific experts for different data types
2. **Traditional Transformer**: Standard transformer architecture for multimodal data

### Key Features

- 📊 **Image-Text Matching Task**: Binary classification of whether image and text match
- 🔧 **Modular Architecture**: Separate encoders for images (ViT) and text (Transformer)
- 📈 **Comprehensive Evaluation**: Performance, efficiency, and computational metrics
- 🎨 **Visualization**: Training curves, efficiency comparisons, and performance plots
- ⚡ **FLOPs Analysis**: Detailed computational complexity comparison

## 🏗️ Architecture

### Data Pipeline

- **Dummy Dataset Creation**: Automatically generates synthetic images and captions
- **Image Processing**: ViT-style patch extraction and encoding
- **Text Processing**: Character-level tokenization and transformer encoding
- **Negative Sampling**: Automatic generation of mismatched image-text pairs

### Models

- **MoT Architecture**: Modality-specific attention and feed-forward layers
- **Traditional Architecture**: Shared parameters across modalities
- **Classification Head**: Binary classifier for match/no-match prediction

### Evaluation Metrics

- **Performance**: Accuracy, F1-score, AUC
- **Efficiency**: FLOPs, inference time, memory usage
- **Training**: Convergence speed, training time

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mixture-of-transformers-case

# Install dependencies
pip install -e .
```

### Run Experiment

```bash
# Full experiment (training + evaluation)
python -m src.experiment

# Quick test run (3 epochs)
python -m src.experiment --quick-run

# Only profile models (no training)
python -m src.experiment --no-training

# Custom configuration
python -m src.experiment --epochs 20 --batch-size 32 --learning-rate 1e-4
```

### Command Line Options

```
--experiment-name TEXT    Experiment name (default: mot_vs_transformer)
--output-dir TEXT         Output directory (default: experiments)
--epochs INT             Number of training epochs (default: 10)
--batch-size INT         Batch size (default: 16)
--learning-rate FLOAT    Learning rate (default: 1e-4)
--device TEXT            Device: auto, cpu, cuda (default: auto)
--seed INT               Random seed (default: 42)
--quick-run              Quick test with 3 epochs
--no-training            Skip training, only profile models
```

## 📁 Project Structure

```
src/
├── __init__.py          # Package initialization
├── config.py            # Configuration dataclasses
├── data.py              # Dataset and data loading
├── encoders.py          # Image and text encoders
├── models.py            # MoT and Traditional models
├── training.py          # Training and validation logic
├── evaluation.py        # Model comparison and visualization
├── utils.py             # Utility functions (FLOPs, profiling)
└── experiment.py        # Main experiment script
```

## 🔧 Customization

### Modify Model Architecture

Edit `src/config.py` to change model parameters:

```python
@dataclass
class ModelConfig:
    hidden_dim: int = 512      # Model dimension
    n_heads: int = 8           # Number of attention heads
    n_layers: int = 6          # Number of transformer layers
    dropout: float = 0.1       # Dropout rate
```

### Use Real Dataset

Replace the dummy dataset in `src/data.py`:

```python
def get_dataloaders(config):
    # Replace create_dummy_dataset() with your real data loading
    train_dataset = YourImageTextDataset(...)
    val_dataset = YourImageTextDataset(...)
    # ... rest of the function
```

### Add New Metrics

Extend the evaluation in `src/evaluation.py`:

```python
def _calculate_custom_metrics(self, model, dataloader):
    # Add your custom evaluation metrics
    pass
```

## 📈 Output Files

After running an experiment, you'll find these files in `experiments/{experiment_name}/`:

## 🧪 Example Results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📚 References

- [Mixture-of-Transformers Paper](https://arxiv.org/abs/2411.04996)
- [Official MoT Repository](https://github.com/facebookresearch/Mixture-of-Transformers)

## 📄 License

This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.
