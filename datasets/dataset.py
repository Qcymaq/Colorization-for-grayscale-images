import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class ColorizationDataset(Dataset):
    def __init__(self, data_root="datasets/", image_size=1024):
        """
        Args:
            data_root (str): Path to the dataset directory.
            image_size (int): Size to which images will be resized.
        """
        self.data = []
        self.data_root = data_root
        self.image_size = image_size

        # Load prompts from the JSON file
        prompts_path = os.path.join(self.data_root, "prompts.json")
        with open(prompts_path, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

        # Define transformations for grayscale and color images
        self.transform_grayscale = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

        self.transform_color = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        grayscale_filename = item['grayscale']
        color_filename = item['color']
        prompt = item['prompt']

        # Construct full paths to grayscale and color images
        grayscale_path = os.path.join(self.data_root, grayscale_filename)
        color_path = os.path.join(self.data_root, color_filename)

        # Load grayscale and color images
        grayscale_image = Image.open(grayscale_path).convert("RGB")  # Ensure 3 channels
        color_image = Image.open(color_path).convert("RGB")

        # Apply transformations
        grayscale_tensor = self.transform_grayscale(grayscale_image)  # [3, 1024, 1024]
        color_tensor = self.transform_color(color_image)  # [3, 1024, 1024]

        # Convert tensors to numpy arrays for consistency with SDXLDataset
        grayscale_np = grayscale_tensor.numpy()
        color_np = color_tensor.numpy()

        # Normalize grayscale images to [0, 1]
        grayscale_np = grayscale_np.astype(np.float32) / 255.0

        # Normalize color images to [-1, 1]
        color_np = color_np.astype(np.float32) / 127.5 - 1.0

        # Convert numpy arrays back to tensors
        grayscale_tensor = torch.from_numpy(grayscale_np)
        color_tensor = torch.from_numpy(color_np)

        return {
            "jpg": color_tensor,  # Color image (Ground Truth)
            "txt": prompt,  # Description prompt
            "hint": grayscale_tensor,  # Grayscale image
            "original_size_as_tuple": torch.tensor([self.image_size, self.image_size]),
            "crop_coords_top_left": torch.tensor([0, 0]),
            "aesthetic_score": torch.tensor([6.0]),
            "target_size_as_tuple": torch.tensor([self.image_size, self.image_size])
        }