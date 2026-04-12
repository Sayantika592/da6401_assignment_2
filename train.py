"""Training entrypoint
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Hyperparameters
EPOCHS = 50
EPOCHS_SEG = 60
BATCH_SIZE = 16
LR = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 4

wandb.init(
    project="oxford_pet_multitask",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "val_split": VAL_SPLIT,
        "optimizer": "Adam",
        "image_size": 224,
        "normalization": "ImageNet",
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = OxfordIIITPetDataset("data", transform)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Train set: {train_size}, Val set: {val_size}")


# Helper: log sample prediction images
def _unnorm(img_tensor):
    """Un-normalise a single CHW tensor back to 0-1 for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


def log_cls_samples(model, loader, tag, num=8):
    """Log a grid of classified images with predicted labels."""
    model.eval()
    images_logged = []
    with torch.no_grad():
        for img, label, bbox, mask in loader:
            img = img.to(device)
            out = model(img)
            _, preds = torch.max(out, 1)
            for i in range(min(num - len(images_logged), img.size(0))):
                np_img = _unnorm(img[i]).permute(1, 2, 0).numpy()
                caption = f"pred={preds[i].item()}, gt={label[i].item()}"
                images_logged.append(wandb.Image(np_img, caption=caption))
            if len(images_logged) >= num:
                break
    wandb.log({f"{tag}/sample_predictions": images_logged})


def log_loc_samples(model, loader, tag, num=8):
    """Log images with predicted vs GT bounding boxes."""
    model.eval()
    images_logged = []
    with torch.no_grad():
        for img, label, bbox, mask in loader:
            img = img.to(device)
            pred_bbox = model(img).cpu()
            for i in range(min(num - len(images_logged), img.size(0))):
                np_img = _unnorm(img[i]).permute(1, 2, 0).numpy()
                fig, ax = plt.subplots(1, figsize=(3, 3))
                ax.imshow(np_img)
                # GT box (green)
                gt = bbox[i]
                gx1, gy1 = gt[0]-gt[2]/2, gt[1]-gt[3]/2
                ax.add_patch(patches.Rectangle((gx1, gy1), gt[2], gt[3],
                             edgecolor='green', fill=False, lw=2, label='GT'))
                # Pred box (red)
                pb = pred_bbox[i]
                px1, py1 = pb[0]-pb[2]/2, pb[1]-pb[3]/2
                ax.add_patch(patches.Rectangle((px1, py1), pb[2], pb[3],
                             edgecolor='red', fill=False, lw=2, label='Pred'))
                ax.legend(fontsize=6)
                ax.axis('off')
                fig.tight_layout()
                images_logged.append(wandb.Image(fig))
                plt.close(fig)
            if len(images_logged) >= num:
                break
    wandb.log({f"{tag}/bbox_predictions": images_logged})


def log_seg_samples(model, loader, tag, num=8):
    """Log images with predicted segmentation masks."""
    model.eval()
    images_logged = []
    with torch.no_grad():
        for img, label, bbox, mask in loader:
            img = img.to(device)
            out = model(img)
            pred_mask = torch.argmax(out, dim=1).cpu()
            for i in range(min(num - len(images_logged), img.size(0))):
                np_img = _unnorm(img[i]).permute(1, 2, 0).numpy()
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(np_img); axes[0].set_title("Input")
                axes[1].imshow(mask[i].numpy(), cmap='tab10', vmin=0, vmax=2); axes[1].set_title("GT Mask")
                axes[2].imshow(pred_mask[i].numpy(), cmap='tab10', vmin=0, vmax=2); axes[2].set_title("Pred Mask")
                for ax in axes: ax.axis('off')
                fig.tight_layout()
                images_logged.append(wandb.Image(fig))
                plt.close(fig)
            if len(images_logged) >= num:
                break
    wandb.log({f"{tag}/seg_predictions": images_logged})


# Task 1: Classification
print("Training classifier...")
classifier = VGG11Classifier(num_classes=37).to(device)
cls_loss_fn = nn.CrossEntropyLoss()
cls_optimizer = optim.Adam(classifier.parameters(), lr=LR)
best_cls_loss = float('inf')

for epoch in range(EPOCHS):
    # Training
    classifier.train()
    total_loss, correct, total = 0, 0, 0
    for img, label, bbox, mask in train_loader:
        img, label = img.to(device), label.to(device)
        out = classifier(img)
        loss = cls_loss_fn(out, label)

        cls_optimizer.zero_grad()
        loss.backward()
        cls_optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(out, 1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # Validation
    classifier.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for img, label, bbox, mask in val_loader:
            img, label = img.to(device), label.to(device)
            out = classifier(img)
            loss = cls_loss_fn(out, label)
            val_loss += loss.item()
            _, preds = torch.max(out, 1)
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    wandb.log({
        "cls/train_loss": train_loss, "cls/train_acc": train_acc,
        "cls/val_loss": val_loss, "cls/val_acc": val_acc,
        "epoch": epoch
    })
    print(f"[Cls] E{epoch+1} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
    
    if val_loss < best_cls_loss:
        best_cls_loss = val_loss
        print(f"  --> Saving best classifier (val_loss: {best_cls_loss:.4f})")
        torch.save(classifier.state_dict(), "checkpoints/classifier.pth")

log_cls_samples(classifier, val_loader, "cls")
classifier.load_state_dict(torch.load("checkpoints/classifier.pth"))
encoder_weights = classifier.encoder.state_dict()

# Task 2: Localization (MSE + IoU loss)
print("Training localizer...")
localizer = VGG11Localizer().to(device)
mse_loss_fn = nn.MSELoss()
iou_loss_fn = IoULoss(reduction="mean")
loc_optimizer = optim.Adam(localizer.parameters(), lr=LR)
best_loc_loss = float('inf')

for epoch in range(EPOCHS):
    # Training
    localizer.train()
    t_loss, t_mse, t_iou, n_batches = 0, 0, 0, 0
    for img, label, bbox, mask in train_loader:
        img, bbox = img.to(device), bbox.to(device)
        pred_bbox = localizer(img)
        loss_mse = mse_loss_fn(pred_bbox, bbox)
        loss_iou = iou_loss_fn(pred_bbox, bbox)
        loss = loss_mse + loss_iou

        loc_optimizer.zero_grad()
        loss.backward()
        loc_optimizer.step()

        t_loss += loss.item()
        t_mse += loss_mse.item()
        t_iou += loss_iou.item()
        n_batches += 1

    train_loss = t_loss / n_batches
    train_mse = t_mse / n_batches
    train_iou = t_iou / n_batches

    # Validation
    localizer.eval()
    v_loss, v_mse, v_iou, v_batches = 0, 0, 0, 0
    with torch.no_grad():
        for img, label, bbox, mask in val_loader:
            img, bbox = img.to(device), bbox.to(device)
            pred_bbox = localizer(img)
            loss_mse = mse_loss_fn(pred_bbox, bbox)
            loss_iou = iou_loss_fn(pred_bbox, bbox)
            v_loss += (loss_mse + loss_iou).item()
            v_mse += loss_mse.item()
            v_iou += loss_iou.item()
            v_batches += 1

    val_loss = v_loss / v_batches
    val_mse = v_mse / v_batches
    val_iou = v_iou / v_batches

    wandb.log({
        "loc/train_loss": train_loss, "loc/train_mse": train_mse, "loc/train_iou": train_iou,
        "loc/val_loss": val_loss, "loc/val_mse": val_mse, "loc/val_iou": val_iou,
        "epoch": epoch
    })
    print(f"[Loc] E{epoch+1} | train={train_loss:.4f} (mse={train_mse:.4f} iou={train_iou:.4f}) | val={val_loss:.4f}")

    if val_loss < best_loc_loss:
        best_loc_loss = val_loss
        print(f"  --> Saving best localizer (val_loss: {best_loc_loss:.4f})")
        torch.save(localizer.state_dict(), "checkpoints/localizer.pth")

log_loc_samples(localizer, val_loader, "loc")

# Task 3: Segmentation
print("="*60); print("Training U-Net segmenter..."); print("="*60)

segmenter = VGG11UNet(num_classes=3).to(device)
# Transfer encoder weights from trained classifier
segmenter.encoder.load_state_dict(encoder_weights)
print("Loaded pretrained encoder from classifier into segmenter.")

# Estimate class weights
print("Computing class weights...")
class_counts = torch.zeros(3)
for i, (img, label, bbox, mask) in enumerate(train_loader):
    for c in range(3):
        class_counts[c] += (mask == c).sum().item()
    if i >= 20: break
total_px = class_counts.sum()
class_weights = total_px / (3.0 * class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * 3.0
print(f"Class weights: {class_weights}")

seg_loss_fn   = nn.CrossEntropyLoss(weight=class_weights.to(device))
seg_optimizer = optim.AdamW(segmenter.parameters(), lr=1e-3, weight_decay=1e-4)
seg_scheduler = optim.lr_scheduler.CosineAnnealingLR(seg_optimizer, T_max=EPOCHS_SEG, eta_min=1e-6)
best_seg_dice = 0.0

for epoch in range(EPOCHS_SEG):
    segmenter.train()
    t_loss, t_correct, t_total, n_batches = 0, 0, 0, 0
    for img, label, bbox, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        out = segmenter(img)
        loss = seg_loss_fn(out, mask)
        seg_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(segmenter.parameters(), max_norm=5.0)
        seg_optimizer.step()
        t_loss += loss.item()
        pred_mask = torch.argmax(out, dim=1)
        t_correct += (pred_mask == mask).sum().item()
        t_total += mask.numel(); n_batches += 1
    seg_scheduler.step()
    train_loss    = t_loss / n_batches
    train_pix_acc = t_correct / t_total

    segmenter.eval()
    v_loss, v_correct, v_total, v_dice, v_batches = 0, 0, 0, 0, 0
    with torch.no_grad():
        for img, label, bbox, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            out = segmenter(img)
            v_loss += seg_loss_fn(out, mask).item()
            pred_mask = torch.argmax(out, dim=1)
            v_correct += (pred_mask == mask).sum().item()
            v_total += mask.numel()
            dice_batch = 0
            for c in range(3):
                p = (pred_mask == c).float(); g = (mask == c).float()
                dice_batch += (2.0*(p*g).sum()+1e-6) / (p.sum()+g.sum()+1e-6)
            v_dice += (dice_batch / 3).item()
            v_batches += 1
    val_loss    = v_loss / v_batches
    val_pix_acc = v_correct / v_total
    val_dice    = v_dice / v_batches

    wandb.log({"seg/train_loss": train_loss, "seg/train_pix_acc": train_pix_acc,
               "seg/val_loss": val_loss, "seg/val_pix_acc": val_pix_acc,
               "seg/val_dice": val_dice, "seg/lr": seg_scheduler.get_last_lr()[0], "epoch": epoch})
    print(f"[Seg] E{epoch+1} | trn_loss={train_loss:.4f} pix={train_pix_acc:.4f} "
          f"| val_loss={val_loss:.4f} pix={val_pix_acc:.4f} dice={val_dice:.4f}")

    if val_dice > best_seg_dice:
        best_seg_dice = val_dice
        print(f"  --> best segmenter (dice={best_seg_dice:.4f})")
        torch.save(segmenter.state_dict(), "checkpoints/unet.pth")

log_seg_samples(segmenter, val_loader, "seg")
print(f"Final segmenter macro-Dice: {compute_macro_dice(segmenter, val_loader):.4f}")

wandb.finish()
print("Training complete. Checkpoints saved.")
