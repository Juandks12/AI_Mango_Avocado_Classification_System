#!/usr/bin/env python3
# train_multi.py
"""
Entrenador para datasets Roboflow COCO (carpetas separadas como aguacate.v1i.coco, mango.v1i.coco).
Soporta: fasterrcnn, ssdlite (MobileNetV3 SSDLite), effdet (opcional).
"""

import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import torchvision

# Optional libs
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False

try:
    from effdet import create_model as effdet_create_model
    HAS_EFFDET = True
except Exception:
    HAS_EFFDET = False

# -------------------- CONFIG --------------------
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 8
DEFAULT_BATCH = 2
DEFAULT_LR = 0.005
DEFAULT_NUM_WORKERS = 4
USE_PRETRAINED = True
USE_MIXED_PRECISION = True
PIN_MEMORY = True
SAVE_EPOCH_CHECKPOINT = True
# ------------------------------------------------


# -------------------- HELPERS --------------------
def find_coco_json(root: Path):
    for sub in ["train", "valid", "val", "test", "."]:
        p = root / sub
        if p.exists():
            for f in p.iterdir():
                if f.suffix.lower() == ".json":
                    try:
                        j = json.loads(f.read_text(encoding="utf8"))
                        if "images" in j and "annotations" in j and "categories" in j:
                            return f
                    except Exception:
                        continue

    for f in root.rglob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf8"))
            if "images" in j and "annotations" in j and "categories" in j:
                return f
        except Exception:
            continue
    return None


class CocoRoboflowDataset(Dataset):
    def __init__(self, json_path: Path, images_dir: Path, transforms=None):
        self.coco = COCO(str(json_path))
        self.ids = sorted(self.coco.getImgIds())
        self.images_dir = Path(images_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        candidates = [
            self.images_dir / img_info['file_name'],
            self.images_dir / Path(img_info['file_name']).name,
            self.images_dir.parent / img_info['file_name'],
            self.images_dir.parent / Path(img_info['file_name']).name
        ]

        img_path = None
        for c in candidates:
            if c.exists():
                img_path = c
                break

        if img_path is None:
            matches = list(self.images_dir.rglob(Path(img_info['file_name']).name))
            img_path = matches[0] if matches else None

        if img_path is None:
            raise FileNotFoundError(f"Image file {img_info['file_name']} not found under {self.images_dir}")

        img = Image.open(str(img_path)).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x,y,w_box,h_box = a['bbox']
            x2, y2 = x + w_box, y + h_box

            x = max(0, x); y = max(0, y)
            x2 = min(x2, img.width); y2 = min(y2, img.height)

            if x2 <= x or y2 <= y:
                continue

            boxes.append([x,y,x2,y2])
            labels.append(a['category_id'])
            areas.append(a.get('area', w_box*h_box))
            iscrowd.append(a.get('iscrowd', 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info['id']]),
            "area": areas,
            "iscrowd": iscrowd
        }

        img_t = T.ToTensor()(img)
        if self.transforms:
            img_t = self.transforms(img_t)

        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# -------------------- MODELOS --------------------
def build_fasterrcnn(num_classes, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_ssdlite(num_classes, pretrained=True):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights="DEFAULT" if pretrained else None
    )
    import torch.nn as nn
    head = model.head
    in_ch = head.classification_head.conv[0].in_channels
    num_anchors = head.classification_head.num_anchors
    new_out = num_anchors * num_classes
    head.classification_head.conv[-1] = nn.Conv2d(in_ch, new_out, 3, padding=1)
    head.classification_head.num_classes = num_classes
    return model


def build_effdet(num_classes, effdet_name="tf_efficientdet_d1"):
    if not HAS_EFFDET:
        raise RuntimeError("effdet/timm not installed")
    return effdet_create_model(effdet_name, bench_task='train',
                               num_classes=num_classes, pretrained=True)


# -------------------- TRAIN --------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, max_epochs, scaler=None):

    desc = f"Epoch {epoch}/{max_epochs} - train"
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=desc)

    model.train()
    running_loss = 0.0

    for i, (imgs, targets) in pbar:
        imgs = [img.to(device) for img in imgs]
        targs = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                loss_dict = model(imgs, targs)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(imgs, targs)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"avg_loss": running_loss / (i+1)})

    return running_loss / len(data_loader)


@torch.no_grad()
def eval_val_loss(model, val_loader, device, epoch, max_epochs, scaler=None):

    desc = f"Epoch {epoch}/{max_epochs} - val_loss"
    pbar = tqdm(val_loader, total=len(val_loader), desc=desc)

    model.eval()
    total_loss, n = 0.0, 0

    for imgs, targets in pbar:
        imgs = [img.to(device) for img in imgs]
        targs = [{k: v.to(device) for k,v in t.items()} for t in targets]

        if scaler:
            with torch.cuda.amp.autocast():
                loss_dict = model(imgs, targs)
                loss = sum(loss_dict.values())
        else:
            loss_dict = model(imgs, targs)
            loss = sum(loss_dict.values())

        total_loss += loss.item()
        n += 1

    return total_loss / n


# -------------------- MAIN --------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="fasterrcnn",
                        choices=("fasterrcnn","ssdlite","effdet"))
    parser.add_argument("--effdet_name", type=str, default="tf_efficientdet_d1")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--output_root", type=str, default="runs")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    model_name = args.model
    max_epochs, patience = args.max_epochs, args.patience

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[i] Device:", device)

    # find jsons
    train_json = find_coco_json(data_root / "train") or find_coco_json(data_root)
    val_json = find_coco_json(data_root / "valid") or find_coco_json(data_root / "val")

    if train_json is None:
        raise SystemExit("No COCO JSON found")

    print("[i] train json:", train_json)
    print("[i] val json:", val_json if val_json else "None")

    train_imgs_dir = train_json.parent
    val_imgs_dir = val_json.parent if val_json else None

    # num_classes
    coco_dict = json.loads(train_json.read_text(encoding="utf8"))
    max_cat = max([c['id'] for c in coco_dict['categories']])
    num_classes = max_cat + 1
    print(f"[i] num_classes = {num_classes}")

    # loaders
    train_ds = CocoRoboflowDataset(train_json, train_imgs_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=PIN_MEMORY)

    val_loader = None
    if val_json:
        val_ds = CocoRoboflowDataset(val_json, val_imgs_dir)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn,
                                pin_memory=PIN_MEMORY)

    # build model
    if model_name == "fasterrcnn":
        model = build_fasterrcnn(num_classes, USE_PRETRAINED)
    elif model_name == "ssdlite":
        model = build_ssdlite(num_classes, USE_PRETRAINED)
    else:
        model = build_effdet(num_classes, args.effdet_name)

    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None

    # dirs
    dataset_name = data_root.name
    save_base = Path(args.output_root) / dataset_name / model_name
    ckpt_dir = save_base / "checkpoints"
    save_base.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_base / "best.pth"

    best_score = None
    epochs_no_improve = 0

    print(f"[i] Starting training: {dataset_name} - {model_name}")

    # -------- TRAINING LOOP --------
    for epoch in range(1, max_epochs+1):

        train_loss = train_one_epoch(
            model, optimizer, train_loader, device,
            epoch, max_epochs, scaler=scaler
        )

        if SAVE_EPOCH_CHECKPOINT:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

        if val_loader:
            val_loss = eval_val_loss(
                model, val_loader, device,
                epoch, max_epochs, scaler=scaler
            )
            current_score = -val_loss
            print(f"[epoch {epoch}] val_loss={val_loss:.4f} -> score={current_score:.6f}")
        else:
            current_score = -train_loss

        if best_score is None or current_score > best_score:
            best_score = current_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"[epoch {epoch}] New BEST saved ({best_path})")
        else:
            epochs_no_improve += 1
            print(f"[epoch {epoch}] No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("[i] Early stopping.")
            break

    print("[i] Training finished.")
    print("[i] Best model:", best_path)


if __name__ == "__main__":
    main()
