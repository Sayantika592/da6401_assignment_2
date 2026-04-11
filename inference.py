"""Inference and evaluation
"""
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A

from models.multitask import MultiTaskPerceptionModel

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = MultiTaskPerceptionModel()
model.to(device)
model.eval()

# transform (must match training normalization)
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def infer(image_path):
    """Run multi-task inference on a single image."""
    image = Image.open(image_path).convert("RGB")
    orig = image.copy()

    # preprocess
    image_np = np.array(image)
    augmented = transform(image=image_np)
    transformed_np = augmented["image"]
    img = torch.from_numpy(transformed_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # forward
    with torch.no_grad():
        outputs = model(img)

    cls_out = outputs["classification"]
    bbox_out = outputs["localization"]
    seg_out = outputs["segmentation"]

    # classification
    pred_class = torch.argmax(cls_out, dim=1).item()

    # bbox in pixel space (already scaled to 224x224)
    bbox = bbox_out[0].cpu().numpy()  # [xc, yc, w, h]

    # segmentation
    seg_mask = torch.argmax(seg_out[0], dim=0).cpu().numpy()

    return orig, pred_class, bbox, seg_mask


def visualize(image, bbox, mask):
    """Visualize bounding box and segmentation mask."""
    img_np = np.array(image.resize((224, 224)))

    xc, yc, bw, bh = bbox  # already in pixel coords for 224x224

    # Convert center format to corner format
    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    plt.figure(figsize=(10, 5))

    # image + bbox
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Bounding Box")
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                       edgecolor='red', fill=False, linewidth=2)
    )

    # segmentation
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Segmentation")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        # Default to taking a random image from the dataset if no argument is provided
        default_dir = "data/images"
        if os.path.exists(default_dir):
            images = [f for f in os.listdir(default_dir) if f.endswith('.jpg')]
            if images:
                import random
                sample_image = os.path.join(default_dir, random.choice(images))
                print(f"No image provided. Using random sample: {sample_image}")
                sys.argv.append(sample_image)
            else:
                print("Usage: python inference.py <path_to_image>")
                sys.exit(1)
        else:
            print("Usage: python inference.py <path_to_image>")
            sys.exit(1)
            
    img_path = sys.argv[1]
    print(f"Running inference on {img_path}...")
    
    try:
        orig_img, cls_pred, bbox_pred, mask_pred = infer(img_path)
        print(f"Predicted Class ID: {cls_pred}")
        print(f"Predicted Bounding Box [xc, yc, w, h]: {bbox_pred}")
        visualize(orig_img, bbox_pred, mask_pred)
    except Exception as e:
        print(f"Error during inference: {e}")