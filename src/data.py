"""
Data loading and preprocessing for Image-Text Matching
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class ImageTextMatchingDataset(Dataset):
    """Dataset for image-text matching task"""

    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        max_text_length: int = 128,
        image_size: int = 224,
        negative_ratio: float = 1.0,
        is_training: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.max_text_length = max_text_length
        self.negative_ratio = negative_ratio
        self.is_training = is_training

        # Load captions
        with open(captions_file, "r") as f:
            self.captions_data = json.load(f)

        # Build image-text pairs
        self.positive_pairs = []
        for img_id, captions in self.captions_data.items():
            for caption in captions:
                self.positive_pairs.append((img_id, caption, 1))  # 1 for match

        # Generate negative pairs
        self.negative_pairs = self._generate_negative_pairs()

        # Combine all pairs
        self.all_pairs = self.positive_pairs + self.negative_pairs

        # Image transforms
        if is_training:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _generate_negative_pairs(self) -> List[Tuple[str, str, int]]:
        """Generate negative image-text pairs"""
        negative_pairs = []
        img_ids = list(self.captions_data.keys())
        all_captions = []

        for captions in self.captions_data.values():
            all_captions.extend(captions)

        n_negatives = int(len(self.positive_pairs) * self.negative_ratio)

        for _ in range(n_negatives):
            # Random image
            img_id = random.choice(img_ids)
            # Random caption from different image
            other_img_id = random.choice(img_ids)
            while other_img_id == img_id:
                other_img_id = random.choice(img_ids)

            wrong_caption = random.choice(self.captions_data[other_img_id])
            negative_pairs.append((img_id, wrong_caption, 0))  # 0 for no match

        return negative_pairs

    def __len__(self) -> int:
        return len(self.all_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id, caption, label = self.all_pairs[idx]

        # Load image
        img_path = self.images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / f"{img_id}.png"

        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        # Tokenize text (simple word-level tokenization)
        text_tokens = self._tokenize_text(caption)

        return {
            "image": image,
            "text": text_tokens,
            "label": torch.tensor(label, dtype=torch.long),
            "img_id": img_id,
            "caption": caption,
        }

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple word-level tokenization"""
        # Convert to lowercase and split
        words = text.lower().split()

        # Simple vocabulary mapping (in practice, use proper tokenizer)
        # For now, use character-level encoding as a placeholder
        text_chars = list(text.lower())[: self.max_text_length]

        # Pad to max length
        if len(text_chars) < self.max_text_length:
            text_chars.extend(["<pad>"] * (self.max_text_length - len(text_chars)))

        # Convert to indices (simple ASCII mapping)
        char_indices = []
        for char in text_chars:
            if char == "<pad>":
                char_indices.append(0)
            else:
                char_indices.append(ord(char) % 256)  # Simple mapping

        return torch.tensor(char_indices, dtype=torch.long)


def create_dummy_dataset(
    output_dir: str, num_images: int = 1000, captions_per_image: int = 3
) -> None:
    """Create a dummy dataset for testing"""
    import os
    from PIL import Image
    import numpy as np

    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/val/images", exist_ok=True)

    # Sample captions
    sample_captions = [
        "A dog running in the park",
        "A cat sitting on a chair",
        "A car driving on the road",
        "A person walking down the street",
        "A bird flying in the sky",
        "A flower blooming in the garden",
        "A child playing with toys",
        "A tree swaying in the wind",
    ]

    def create_split(split_name: str, n_images: int):
        captions_dict = {}

        for i in range(n_images):
            img_id = f"{split_name}_img_{i:04d}"

            # Create dummy image
            dummy_img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            dummy_img.save(f"{output_dir}/{split_name}/images/{img_id}.jpg")

            # Assign random captions
            captions = random.sample(sample_captions, captions_per_image)
            captions_dict[img_id] = captions

        # Save captions file
        with open(f"{output_dir}/{split_name}/captions.json", "w") as f:
            json.dump(captions_dict, f, indent=2)

    # Create train and validation splits
    create_split("train", int(num_images * 0.8))
    create_split("val", int(num_images * 0.2))

    print(f"Created dummy dataset in {output_dir}")


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""

    # Create dummy dataset if it doesn't exist
    if not Path(config.data.data_root).exists():
        print("Creating dummy dataset...")
        create_dummy_dataset(config.data.data_root)

    # Create datasets
    train_dataset = ImageTextMatchingDataset(
        images_dir=f"{config.data.data_root}/{config.data.train_images_dir}",
        captions_file=f"{config.data.data_root}/{config.data.train_captions_file}",
        max_text_length=config.data.max_text_length,
        image_size=config.data.image_size,
        negative_ratio=config.data.negative_ratio,
        is_training=True,
    )

    val_dataset = ImageTextMatchingDataset(
        images_dir=f"{config.data.data_root}/{config.data.val_images_dir}",
        captions_file=f"{config.data.data_root}/{config.data.val_captions_file}",
        max_text_length=config.data.max_text_length,
        image_size=config.data.image_size,
        negative_ratio=config.data.negative_ratio,
        is_training=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
