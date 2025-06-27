#!/usr/bin/env python3
"""
Test script to verify the MoT framework setup
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test all module imports"""
    print("🔍 Testing imports...")

    try:
        # Test basic imports first
        import torchvision

        print(f"✅ torchvision {torchvision.__version__} imported successfully")

        from src.config import ExperimentConfig

        print("✅ Config imported successfully")

        from src.data import ImageTextMatchingDataset, create_dummy_dataset

        print("✅ Data modules imported successfully")

        from src.encoders import TextEncoder, ImageEncoder, MultiModalEncoder

        print("✅ Encoders imported successfully")

        from src.models import MoTModel, TraditionalTransformerModel

        print("✅ Models imported successfully")

        from src.utils import FLOPsCounter, ModelProfiler, set_seed

        print("✅ Utils imported successfully")

        from src.training import Trainer

        print("✅ Training imported successfully")

        from src.evaluation import ModelComparator

        print("✅ Evaluation imported successfully")

        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("\n🔍 Testing configuration...")

    try:
        from src.config import ExperimentConfig

        config = ExperimentConfig()
        print(f"✅ Config created: {config.experiment_name}")
        print(f"   - Model hidden_dim: {config.model.hidden_dim}")
        print(f"   - Training epochs: {config.training.max_epochs}")
        print(f"   - Data batch_size: {config.data.batch_size}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_models():
    """Test model creation"""
    print("\n🔍 Testing model creation...")

    try:
        from src.config import ExperimentConfig, ModelConfig
        from src.models import MoTModel, TraditionalTransformerModel

        config = ModelConfig()
        print(
            f"   Model config: {config.n_layers} layers, {config.hidden_dim} hidden_dim"
        )

        # Test MoT model
        mot_model = MoTModel(config)
        mot_params = sum(p.numel() for p in mot_model.parameters())
        print(f"✅ MoT model created with {mot_params:,} parameters")

        # Test Traditional model
        trad_model = TraditionalTransformerModel(config)
        trad_params = sum(p.numel() for p in trad_model.parameters())
        print(f"✅ Traditional model created with {trad_params:,} parameters")

        # Parameter comparison
        param_diff = mot_params - trad_params
        print(
            f"   Parameter difference: {param_diff:,} ({param_diff/trad_params*100:.1f}%)"
        )

        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass"""
    print("\n🔍 Testing forward pass...")

    try:
        from src.config import ModelConfig
        from src.models import MoTModel, TraditionalTransformerModel

        config = ModelConfig()

        # Create sample data
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        text_tokens = torch.randint(0, 256, (batch_size, 128))
        print(f"   Input shapes: images {images.shape}, text {text_tokens.shape}")

        # Test MoT model
        mot_model = MoTModel(config)
        mot_model.eval()
        with torch.no_grad():
            mot_output = mot_model(images, text_tokens)
        print(f"✅ MoT forward pass: {mot_output.shape}")

        # Test Traditional model
        trad_model = TraditionalTransformerModel(config)
        trad_model.eval()
        with torch.no_grad():
            trad_output = trad_model(images, text_tokens)
        print(f"✅ Traditional forward pass: {trad_output.shape}")

        # Check outputs
        if mot_output.shape == trad_output.shape:
            print("✅ Output shapes match!")
        else:
            print(
                f"⚠️  Output shapes differ: MoT {mot_output.shape} vs Traditional {trad_output.shape}"
            )

        return True
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_flops_counter():
    """Test FLOPs counter"""
    print("\n🔍 Testing FLOPs counter...")

    try:
        from src.config import ModelConfig
        from src.models import MoTModel, TraditionalTransformerModel
        from src.utils import FLOPsCounter

        config = ModelConfig()
        flops_counter = FLOPsCounter()

        # Create sample data
        batch_size = 2  # Smaller batch for testing
        images = torch.randn(batch_size, 3, 224, 224)
        text_tokens = torch.randint(0, 256, (batch_size, 128))
        print(f"   Using batch_size={batch_size} for FLOPs counting")

        # Test MoT model FLOPs
        mot_model = MoTModel(config)
        mot_flops = flops_counter.count_flops(mot_model, (images, text_tokens))
        print(f"✅ MoT FLOPs: {mot_flops:,}")

        # Test Traditional model FLOPs
        trad_model = TraditionalTransformerModel(config)
        trad_flops = flops_counter.count_flops(trad_model, (images, text_tokens))
        print(f"✅ Traditional FLOPs: {trad_flops:,}")

        # Calculate reduction/increase
        if trad_flops > 0:
            flops_change = (1 - mot_flops / trad_flops) * 100
            if flops_change > 0:
                print(f"✅ FLOPs reduction: {flops_change:.1f}%")
            else:
                print(
                    f"⚠️  FLOPs increase: {-flops_change:.1f}% (This may indicate an issue)"
                )
        else:
            print("⚠️  Could not calculate FLOPs comparison")

        return True
    except Exception as e:
        print(f"❌ FLOPs counter error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading"""
    print("\n🔍 Testing data loading...")

    try:
        from src.config import ExperimentConfig
        from src.data import create_dummy_dataset, get_dataloaders

        config = ExperimentConfig()
        config.data.batch_size = 4  # Small batch for testing
        config.data.num_workers = 0  # Avoid multiprocessing issues

        # Create dummy dataset
        print("   Creating dummy dataset...")
        create_dummy_dataset("test_data", num_images=20)
        config.data.data_root = "test_data"

        # Get dataloaders
        print("   Creating dataloaders...")
        train_loader, val_loader = get_dataloaders(config)

        print(f"✅ Train dataset: {len(train_loader.dataset)} samples")
        print(f"✅ Val dataset: {len(val_loader.dataset)} samples")

        # Test one batch
        print("   Testing batch loading...")
        batch = next(iter(train_loader))
        print(
            f"✅ Batch shape - Images: {batch['image'].shape}, Text: {batch['text'].shape}"
        )
        print(f"   Labels: {batch['label']}")
        print(f"   Sample caption: '{batch['caption'][0]}'")

        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 MoT Framework Setup Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    tests = [
        test_imports,
        test_config,
        test_models,
        test_forward_pass,
        test_flops_counter,
        test_data_loading,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your MoT framework is ready to use.")
        print("\nNext steps:")
        print("1. Run quick test: python -m src.experiment --quick-run")
        print("2. Run full experiment: python -m src.experiment")
        print("3. Profile only: python -m src.experiment --no-training")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nEven with partial failures, you can try running:")
        print("  python -m src.experiment --quick-run")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
