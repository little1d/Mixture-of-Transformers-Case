"""
Main experiment script for MoT vs Traditional Transformer comparison
"""

import os
import argparse
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import get_dataloaders
from .models import MoTModel, TraditionalTransformerModel
from .training import create_trainer
from .evaluation import ModelComparator
from .utils import set_seed, format_time


def run_experiment(config: ExperimentConfig):
    """Run the complete MoT vs Traditional Transformer experiment"""

    print("🚀 Starting MoT vs Traditional Transformer Experiment")
    print("=" * 60)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Create output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Task: Image-Text Matching")
    print(f"🔧 Device: {config.training.device}")
    print(f"📊 Dataset: Dummy dataset (for demonstration)")

    # Load data
    print("\n📋 Loading data...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {config.data.batch_size}")

    # Initialize models
    print("\n🏗️  Initializing models...")

    # MoT Model
    mot_model = MoTModel(config.model)
    mot_params = sum(p.numel() for p in mot_model.parameters())
    print(f"MoT Model parameters: {mot_params:,}")

    # Traditional Transformer Model
    traditional_model = TraditionalTransformerModel(config.model)
    trad_params = sum(p.numel() for p in traditional_model.parameters())
    print(f"Traditional Model parameters: {trad_params:,}")

    # Training results storage
    results = {}

    if config.compare_models:
        print("\n🏋️  Training Models...")

        # Train MoT Model
        print("\n" + "=" * 50)
        print("Training MoT Model")
        print("=" * 50)

        mot_trainer = create_trainer(
            mot_model, train_loader, val_loader, config, "mot_model"
        )
        mot_results = mot_trainer.train()
        results["mot"] = mot_results

        # Train Traditional Model
        print("\n" + "=" * 50)
        print("Training Traditional Transformer Model")
        print("=" * 50)

        trad_trainer = create_trainer(
            traditional_model, train_loader, val_loader, config, "traditional_model"
        )
        trad_results = trad_trainer.train()
        results["traditional"] = trad_results

        # Model Comparison
        print("\n" + "=" * 50)
        print("Model Comparison & Analysis")
        print("=" * 50)

        comparator = ModelComparator(config, config.output_dir)
        comparison_results = comparator.compare_models(
            mot_model=mot_model,
            traditional_model=traditional_model,
            val_loader=val_loader,
            mot_results=mot_results,
            traditional_results=trad_results,
        )
        results["comparison"] = comparison_results

    else:
        print("\n⚠️  Skipping model training (compare_models=False)")

        # Just profile the models without training
        print("\n📊 Profiling models without training...")
        comparator = ModelComparator(config, config.output_dir)

        # Create dummy results for profiling
        dummy_results = {
            "best_val_acc": 0.0,
            "total_time": 0.0,
            "train_metrics": [],
            "val_metrics": [],
        }

        comparison_results = comparator.compare_models(
            mot_model=mot_model,
            traditional_model=traditional_model,
            val_loader=val_loader,
            mot_results=dummy_results,
            traditional_results=dummy_results,
        )
        results["comparison"] = comparison_results

    print("\n✅ Experiment completed successfully!")
    print(f"📁 All results saved to: {output_dir}")

    return results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="MoT vs Traditional Transformer Experiment"
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (optional)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mot_vs_transformer",
        help="Experiment name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Quick run with reduced epochs for testing",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip training and only profile models",
    )

    args = parser.parse_args()

    # Create configuration
    config = ExperimentConfig()

    # Override with command line arguments
    config.experiment_name = args.experiment_name
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.compare_models = not args.no_training

    # Training configuration
    if args.quick_run:
        config.training.max_epochs = 3
        config.data.batch_size = 8
        print("🚀 Quick run mode enabled (3 epochs, batch_size=8)")
    else:
        config.training.max_epochs = args.epochs
        config.data.batch_size = args.batch_size

    config.training.learning_rate = args.learning_rate

    # Device configuration
    if args.device == "auto":
        config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config.training.device = args.device

    # Adjust configuration for device
    if config.training.device == "cpu":
        config.data.num_workers = 0  # Avoid multiprocessing issues on CPU
        print("⚠️  Running on CPU - reduced num_workers to 0")

    print(
        f"Configuration: {config.training.device}, epochs={config.training.max_epochs}, lr={config.training.learning_rate}"
    )

    # Run experiment
    try:
        results = run_experiment(config)
        print("\n🎉 Experiment completed successfully!")

        if config.compare_models and "comparison" in results:
            # Print key metrics
            comparison = results["comparison"]
            flops_reduction = comparison["efficiency_gains"]["flops_reduction"]
            speedup = comparison["efficiency_gains"]["inference_speedup"]
            acc_diff = comparison["efficiency_gains"]["accuracy_diff"]

            print(f"\n📈 Key Results:")
            print(f"   FLOPs reduction: {flops_reduction:.1f}%")
            print(f"   Inference speedup: {speedup:.2f}x")
            print(f"   Accuracy difference: {acc_diff:+.2f}%")

    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
