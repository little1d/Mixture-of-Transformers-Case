"""
Evaluation and comparison of MoT vs Traditional Transformer
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from .utils import ModelProfiler, format_time, format_number


class ModelComparator:
    """Compare MoT and Traditional Transformer models"""

    def __init__(self, config, output_dir: str = "experiments"):
        self.config = config
        self.output_dir = Path(output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config.training.device)
        self.profiler = ModelProfiler(self.device)

    def compare_models(
        self,
        mot_model: nn.Module,
        traditional_model: nn.Module,
        val_loader: DataLoader,
        mot_results: Dict[str, Any],
        traditional_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two models comprehensively"""

        print("=" * 60)
        print("MODEL COMPARISON REPORT")
        print("=" * 60)

        # Get sample batch for profiling
        sample_batch = next(iter(val_loader))
        sample_images = sample_batch["image"][:4].to(
            self.device
        )  # Use smaller batch for profiling
        sample_text = sample_batch["text"][:4].to(self.device)

        # Profile both models
        print("\nProfiling models...")
        mot_profile = self.profiler.profile_model(
            mot_model, (sample_images, sample_text)
        )
        traditional_profile = self.profiler.profile_model(
            traditional_model, (sample_images, sample_text)
        )

        # Calculate metrics
        comparison_results = {
            "model_comparison": {
                "mot": {
                    "parameters": mot_profile["n_parameters"],
                    "flops": mot_profile["flops"],
                    "inference_time": mot_profile["mean_inference_time"],
                    "throughput": mot_profile["throughput"],
                    "best_accuracy": mot_results["best_val_acc"],
                    "training_time": mot_results["total_time"],
                },
                "traditional": {
                    "parameters": traditional_profile["n_parameters"],
                    "flops": traditional_profile["flops"],
                    "inference_time": traditional_profile["mean_inference_time"],
                    "throughput": traditional_profile["throughput"],
                    "best_accuracy": traditional_results["best_val_acc"],
                    "training_time": traditional_results["total_time"],
                },
            },
            "efficiency_gains": {
                "flops_reduction": (
                    1 - mot_profile["flops"] / traditional_profile["flops"]
                )
                * 100,
                "inference_speedup": traditional_profile["mean_inference_time"]
                / mot_profile["mean_inference_time"],
                "accuracy_diff": mot_results["best_val_acc"]
                - traditional_results["best_val_acc"],
                "training_time_ratio": mot_results["total_time"]
                / traditional_results["total_time"],
            },
        }

        # Print comparison
        self._print_comparison(comparison_results)

        # Create visualizations
        self._create_visualizations(
            mot_results, traditional_results, comparison_results
        )

        # Save results
        self._save_results(comparison_results, mot_results, traditional_results)

        return comparison_results

    def _print_comparison(self, results: Dict[str, Any]):
        """Print detailed comparison results"""
        mot = results["model_comparison"]["mot"]
        trad = results["model_comparison"]["traditional"]
        gains = results["efficiency_gains"]

        print(f"\n{'Metric':<25} {'MoT':<15} {'Traditional':<15} {'Improvement':<15}")
        print("-" * 70)

        # Parameters
        print(
            f"{'Parameters':<25} {format_number(mot['parameters']):<15} "
            f"{format_number(trad['parameters']):<15} {'-':<15}"
        )

        # FLOPs
        flops_reduction = gains["flops_reduction"]
        print(
            f"{'FLOPs':<25} {format_number(mot['flops']):<15} "
            f"{format_number(trad['flops']):<15} {flops_reduction:+.1f}%"
        )

        # Inference time
        speedup = gains["inference_speedup"]
        print(
            f"{'Inference Time':<25} {mot['inference_time']*1000:.1f}ms "
            f"{trad['inference_time']*1000:.1f}ms {speedup:.2f}x faster"
        )

        # Accuracy
        acc_diff = gains["accuracy_diff"]
        print(
            f"{'Accuracy':<25} {mot['best_accuracy']:.2f}% "
            f"{trad['best_accuracy']:.2f}% {acc_diff:+.2f}%"
        )

        # Training time
        time_ratio = gains["training_time_ratio"]
        print(
            f"{'Training Time':<25} {format_time(mot['training_time']):<15} "
            f"{format_time(trad['training_time']):<15} {time_ratio:.2f}x"
        )

        print("\n" + "=" * 70)
        print("SUMMARY:")
        print(f"• MoT reduces FLOPs by {flops_reduction:.1f}%")
        print(f"• MoT is {speedup:.2f}x faster in inference")
        print(f"• Accuracy difference: {acc_diff:+.2f}%")
        if flops_reduction > 0 and acc_diff >= -1:  # Less than 1% accuracy drop
            print("✅ MoT achieves better efficiency with maintained accuracy!")
        elif flops_reduction > 0:
            print(
                f"⚖️  MoT trades {-acc_diff:.1f}% accuracy for {flops_reduction:.1f}% FLOPs reduction"
            )
        print("=" * 70)

    def _create_visualizations(
        self,
        mot_results: Dict[str, Any],
        traditional_results: Dict[str, Any],
        comparison: Dict[str, Any],
    ):
        """Create comparison visualizations"""

        # Training curves
        self._plot_training_curves(mot_results, traditional_results)

        # Efficiency comparison
        self._plot_efficiency_comparison(comparison)

        # Performance metrics
        self._plot_performance_metrics(comparison)

    def _plot_training_curves(self, mot_results: Dict, traditional_results: Dict):
        """Plot training and validation curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Extract metrics
        mot_train = mot_results["train_metrics"]
        mot_val = mot_results["val_metrics"]
        trad_train = traditional_results["train_metrics"]
        trad_val = traditional_results["val_metrics"]

        # Training loss
        ax1.plot(
            [m["epoch"] for m in mot_train],
            [m["loss"] for m in mot_train],
            "b-",
            label="MoT",
            linewidth=2,
        )
        ax1.plot(
            [m["epoch"] for m in trad_train],
            [m["loss"] for m in trad_train],
            "r-",
            label="Traditional",
            linewidth=2,
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation loss
        ax2.plot(
            [m["epoch"] for m in mot_val],
            [m["loss"] for m in mot_val],
            "b-",
            label="MoT",
            linewidth=2,
        )
        ax2.plot(
            [m["epoch"] for m in trad_val],
            [m["loss"] for m in trad_val],
            "r-",
            label="Traditional",
            linewidth=2,
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss")
        ax2.set_title("Validation Loss Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Training accuracy
        ax3.plot(
            [m["epoch"] for m in mot_train],
            [m["accuracy"] for m in mot_train],
            "b-",
            label="MoT",
            linewidth=2,
        )
        ax3.plot(
            [m["epoch"] for m in trad_train],
            [m["accuracy"] for m in trad_train],
            "r-",
            label="Traditional",
            linewidth=2,
        )
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Training Accuracy (%)")
        ax3.set_title("Training Accuracy Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Validation accuracy
        ax4.plot(
            [m["epoch"] for m in mot_val],
            [m["accuracy"] for m in mot_val],
            "b-",
            label="MoT",
            linewidth=2,
        )
        ax4.plot(
            [m["epoch"] for m in trad_val],
            [m["accuracy"] for m in trad_val],
            "r-",
            label="Traditional",
            linewidth=2,
        )
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Validation Accuracy (%)")
        ax4.set_title("Validation Accuracy Comparison")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_efficiency_comparison(self, comparison: Dict[str, Any]):
        """Plot efficiency metrics comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        mot = comparison["model_comparison"]["mot"]
        trad = comparison["model_comparison"]["traditional"]

        # FLOPs comparison
        models = ["MoT", "Traditional"]
        flops = [mot["flops"], trad["flops"]]
        colors = ["skyblue", "lightcoral"]

        bars1 = ax1.bar(models, flops, color=colors)
        ax1.set_ylabel("FLOPs")
        ax1.set_title("Computational Complexity (FLOPs)")
        ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # Add value labels on bars
        for bar, flop in zip(bars1, flops):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{format_number(int(flop))}",
                ha="center",
                va="bottom",
            )

        # Inference time comparison
        inf_times = [
            mot["inference_time"] * 1000,
            trad["inference_time"] * 1000,
        ]  # Convert to ms
        bars2 = ax2.bar(models, inf_times, color=colors)
        ax2.set_ylabel("Inference Time (ms)")
        ax2.set_title("Inference Speed Comparison")

        # Add value labels on bars
        for bar, time_ms in zip(bars2, inf_times):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{time_ms:.1f}ms",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "efficiency_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_performance_metrics(self, comparison: Dict[str, Any]):
        """Plot performance metrics comparison"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        mot = comparison["model_comparison"]["mot"]
        trad = comparison["model_comparison"]["traditional"]

        metrics = ["Accuracy", "FLOPs\n(normalized)", "Inference Time\n(normalized)"]

        # Normalize metrics for comparison
        mot_values = [
            mot["best_accuracy"],
            (mot["flops"] / trad["flops"]) * 100,  # Normalize to traditional
            (mot["inference_time"] / trad["inference_time"])
            * 100,  # Normalize to traditional
        ]

        trad_values = [
            trad["best_accuracy"],
            100,  # Reference baseline
            100,  # Reference baseline
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width / 2, mot_values, width, label="MoT", color="skyblue")
        bars2 = ax.bar(
            x + width / 2, trad_values, width, label="Traditional", color="lightcoral"
        )

        ax.set_ylabel("Score / Normalized Value")
        ax.set_title("Performance Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "performance_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _save_results(
        self,
        comparison: Dict[str, Any],
        mot_results: Dict[str, Any],
        traditional_results: Dict[str, Any],
    ):
        """Save all results to files"""

        # Save comparison results
        with open(self.output_dir / "comparison_results.json", "w") as f:
            json.dump(comparison, f, indent=2)

        # Save detailed results
        results = {
            "comparison": comparison,
            "mot_results": mot_results,
            "traditional_results": traditional_results,
            "config": (
                self.config.__dict__
                if hasattr(self.config, "__dict__")
                else str(self.config)
            ),
        }

        with open(self.output_dir / "full_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Create summary DataFrame
        summary_data = {
            "Model": ["MoT", "Traditional"],
            "Parameters": [
                comparison["model_comparison"]["mot"]["parameters"],
                comparison["model_comparison"]["traditional"]["parameters"],
            ],
            "FLOPs": [
                comparison["model_comparison"]["mot"]["flops"],
                comparison["model_comparison"]["traditional"]["flops"],
            ],
            "Inference Time (ms)": [
                comparison["model_comparison"]["mot"]["inference_time"] * 1000,
                comparison["model_comparison"]["traditional"]["inference_time"] * 1000,
            ],
            "Best Accuracy (%)": [
                comparison["model_comparison"]["mot"]["best_accuracy"],
                comparison["model_comparison"]["traditional"]["best_accuracy"],
            ],
            "Training Time (s)": [
                comparison["model_comparison"]["mot"]["training_time"],
                comparison["model_comparison"]["traditional"]["training_time"],
            ],
        }

        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / "summary.csv", index=False)

        print(f"\nResults saved to {self.output_dir}")
        print(
            f"• Comparison plots: training_curves.png, efficiency_comparison.png, performance_metrics.png"
        )
        print(f"• Data files: comparison_results.json, full_results.json, summary.csv")
