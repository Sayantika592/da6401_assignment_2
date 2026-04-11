"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np
import xml.etree.ElementTree as ET

import albumentations as A

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Directory structure expected:
        root_dir/
            images/          - .jpg images
            annotations/
                trimaps/     - .png trimap masks
                xmls/        - .xml bounding box annotations
                list.txt     - image list with class IDs
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations/trimaps")

        # Build image list, filtering out non-image files
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(".jpg")
        ])

        # Parse the annotations list.txt to get correct class labels
        # Format: Image CLASS-ID SPECIES BREED-ID
        # CLASS-ID: 1-37 class index
        self.label_map = {}
        list_path = os.path.join(root_dir, "annotations", "list.txt")
        if os.path.exists(list_path):
            with open(list_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or len(line) == 0:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        img_name = parts[0]  # e.g., "Abyssinian_1"
                        class_id = int(parts[1])  # 1-37
                        self.label_map[img_name] = class_id - 1  # 0-indexed

        # Filter images to only those with XML annotations
        self.images = [
            img for img in self.images
            if os.path.exists(os.path.join(
                self.root_dir, "annotations/xmls",
                img.replace(".jpg", ".xml")
            ))
        ]

    def __len__(self):
        return len(self.images)

    def parse_bbox(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        return xmin, ymin, xmax, ymax

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # Get original image size before any transforms
        w_img, h_img = image.size

        # Mask
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        mask = np.array(mask) - 1  # trimaps are 1,2,3 -> convert to 0,1,2
        mask = torch.tensor(mask, dtype=torch.long)

        # Label
        # Use the annotations list.txt for correct class mapping
        base_name = img_name.replace(".jpg", "")
        if base_name in self.label_map:
            label = self.label_map[base_name]
        else:
            # Fallback: derive from filename prefix (less reliable)
            # Group by breed name (everything before the last _number)
            parts = base_name.rsplit("_", 1)
            label = 0  # placeholder
        label = torch.tensor(label, dtype=torch.long)

        # Bounding box
        xml_name = img_name.replace(".jpg", ".xml")
        xml_path = os.path.join(self.root_dir, "annotations/xmls", xml_name)
        xmin, ymin, xmax, ymax = self.parse_bbox(xml_path)

        # Convert to [x_center, y_center, width, height] in pixel space
        # (relative to original image, will be scaled after resize)
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
        w = float(xmax - xmin)
        h = float(ymax - ymin)

        # Scale bbox to 224x224 (the resized image dimensions)
        scale_x = 224.0 / w_img
        scale_y = 224.0 / h_img
        xc = xc * scale_x
        yc = yc * scale_y
        w = w * scale_x
        h = h * scale_y

        bbox = torch.tensor([xc, yc, w, h], dtype=torch.float32)

        image_np = np.array(image)
        if self.transform:
            augmented = self.transform(image=image_np)
            image_np = augmented["image"]
            
            # Convert numpy array (H, W, C) to torch tensor (C, H, W)
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            
            # Resize mask to match image
            mask = mask.unsqueeze(0).float()
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), size=(224, 224), mode="nearest"
            )
            mask = mask.squeeze(0).squeeze(0).long()
        else:
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        return image, label, bbox, mask