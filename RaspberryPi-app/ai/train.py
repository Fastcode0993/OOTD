"""
ai/train.py
────────────
참고용 학습 스크립트 (더미).
실제 6만 장 데이터셋이 있을 때 이 구조로 학습한 뒤
`personal_color.pt` 를 생성하면 infer.py 에서 그대로 사용 가능.
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model_loader import build_model, PERSONAL_COLOR_CLASSES, NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 하이퍼파라미터 ──────────────────────────────────────────────────────
BATCH_SIZE     = 32
EPOCHS         = 30
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
SAVE_PATH      = "models/personal_color.pt"


# ── 데이터셋 ────────────────────────────────────────────────────────────
class PersonalColorDataset(Dataset):
    """
    폴더 구조:
        data/train/{color_code}/*.jpg
    예: data/train/spring_warm_light/img_001.jpg
    """

    def __init__(self, root: str, split: str = "train") -> None:
        self.root = Path(root) / split
        self.classes = PERSONAL_COLOR_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            folder = self.root / cls
            if not folder.exists():
                continue
            for img_path in folder.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls]))
            for img_path in folder.glob("*.png"):
                self.samples.append((img_path, self.class_to_idx[cls]))

        logger.info(f"[{split}] {len(self.samples)} samples loaded.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


# ── 학습 루프 ────────────────────────────────────────────────────────────
def train() -> None:
    device = torch.device("cpu")  # 라즈베리파이5 환경

    train_ds = PersonalColorDataset("data", split="train")
    val_ds   = PersonalColorDataset("data", split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(NUM_CLASSES).to(device)

    # 다단계 학습: 먼저 헤드만, 이후 전체 파인튜닝
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ── 3 에폭 이후 전체 레이어 파인튜닝 ──────────────────────────
        if epoch == 4:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
            logger.info("Switched to full fine-tuning.")

        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        scheduler.step()
        n_train = len(train_ds)
        t_acc = train_correct / n_train

        # ── 검증 ──────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
        v_acc = val_correct / len(val_ds)

        logger.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={train_loss/n_train:.4f} | "
            f"train_acc={t_acc:.3f} | val_acc={v_acc:.3f}"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            logger.info(f"  ✓ Saved best model (val_acc={v_acc:.3f})")

    logger.info(f"Training complete. Best val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    train()
