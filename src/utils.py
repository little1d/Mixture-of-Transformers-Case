"""
Utility functions for MoT experiments
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Callable
from functools import wraps


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FLOPsCounter:
    """Simple FLOPs counter for model operations"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_flops = 0
        self.hooks = []

    def count_flops(self, model: nn.Module, input_data: tuple):
        """Count FLOPs for a forward pass"""
        self.reset()

        # Register hooks for different layer types
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._linear_flop_jit)
                self.hooks.append(hook)
            elif isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_flop_jit)
                self.hooks.append(hook)
            elif isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(self._conv_flop_jit)
                self.hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            if isinstance(input_data, (list, tuple)):
                _ = model(*input_data)
            else:
                _ = model(input_data)

        # Clean up hooks
        for hook in self.hooks:
            hook.remove()

        return self.total_flops

    def _linear_flop_jit(self, module, input, output):
        """Count FLOPs for linear layer"""
        input_numel = input[0].numel()
        output_numel = output.numel()
        # Each output element requires input_features multiplications and additions
        self.total_flops += input_numel * module.weight.size(0)

    def _attention_flop_jit(self, module, input, output):
        """Count FLOPs for attention layer (approximate)"""
        seq_len = input[0].size(1)
        embed_dim = module.embed_dim
        # Q, K, V projections + attention computation + output projection
        self.total_flops += 4 * seq_len * embed_dim * embed_dim
        # Attention matrix computation
        self.total_flops += seq_len * seq_len * embed_dim

    def _conv_flop_jit(self, module, input, output):
        """Count FLOPs for conv layer"""
        output_numel = output.numel()
        kernel_flops = (
            module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        )
        self.total_flops += output_numel * kernel_flops


def time_function(func: Callable) -> Callable:
    """Decorator to time function execution"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper


class ModelProfiler:
    """Profile model performance metrics"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.flops_counter = FLOPsCounter()

    def profile_model(
        self,
        model: nn.Module,
        input_data: tuple,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """Profile model with given input"""
        model.eval()
        model = model.to(self.device)

        # Move input data to device
        if isinstance(input_data, (list, tuple)):
            input_data = tuple(
                x.to(self.device) if torch.is_tensor(x) else x for x in input_data
            )
        else:
            input_data = input_data.to(self.device)

        # Count parameters
        n_params = count_parameters(model)

        # Count FLOPs
        flops = self.flops_counter.count_flops(model, input_data)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if isinstance(input_data, (list, tuple)):
                    _ = model(*input_data)
                else:
                    _ = model(input_data)

        # Time inference
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        inference_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                if isinstance(input_data, (list, tuple)):
                    _ = model(*input_data)
                else:
                    _ = model(input_data)

                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()

                end_time = time.time()
                inference_times.append(end_time - start_time)

        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)

        return {
            "n_parameters": n_params,
            "flops": flops,
            "mean_inference_time": mean_time,
            "std_inference_time": std_time,
            "throughput": 1.0 / mean_time if mean_time > 0 else 0,
            "inference_times": inference_times,
        }


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)):
    """Compute accuracy for specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m {seconds % 60:.0f}s"


def format_number(num: int) -> str:
    """Format large numbers with appropriate units"""
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num / 1000:.1f}K"
    elif num < 1000000000:
        return f"{num / 1000000:.1f}M"
    else:
        return f"{num / 1000000000:.1f}B"
