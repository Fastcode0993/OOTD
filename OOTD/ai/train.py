"""
ai/train.py
────────────
계층적 퍼스널 컬러 모델 학습 스크립트.
Stage 1 (4-class 계절) + Stage 2 (3-class 톤) 동시 학습.
데이터 폴더 구조: data/{split}/{personal_color_code}/*.jpg
예: data/train/spring_warm_light/img_001.jpg

학습 후 hierarchical.pt 로 저장.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model_loader import (
    HierarchicalPersonalColorModel,
    PERSONAL_COLOR_CLASSES,
    TONE_LABELS,
    SEASONS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SAVE_PATH    = "models/hierarchical.pt"

# 퍼스널 컬러 코드 → (season_idx, tone_idx) 매핑
_COLOR_TO_LABELS: dict[str, tuple[int, int]] = {
    color: (s_idx, t_idx)
    for s_idx, tones in TONE_LABELS.items()
    for t_idx, color in enumerate(tones)
}


# ── 데이터셋 ──────────────────────────────────────────────────────────────
class PersonalColorDataset(Dataset):
    """
    폴더 구조:
        data/{split}/{color_code}/*.jpg
    예: data/train/spring_warm_light/img_001.jpg
    """

    def __init__(self, root: str, split: str = "train") -> None:
        self.root = Path(root) / split

        # 학습용 transform (augmentation 포함)
        augment = split == "train"
        tf_list = [
            transforms.Resize((224, 224)),
        ]
        if augment:
            tf_list += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            ]
        tf_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(tf_list)

        self.samples: list[tuple[Path, int, int]] = []  # (path, season_idx, tone_idx)
        for color in PERSONAL_COLOR_CLASSES:
            if color not in _COLOR_TO_LABELS:
                continue
            season_idx, tone_idx = _COLOR_TO_LABELS[color]
            folder = self.root / color
            if not folder.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in folder.glob(ext):
                    self.samples.append((p, season_idx, tone_idx))

        logger.info(f"[{split}] {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, season_idx, tone_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, season_idx, tone_idx


# ── 학습 루프 ─────────────────────────────────────────────────────────────
def train() -> None:
    device = torch.device("cpu")

    train_ds = PersonalColorDataset("data", split="train")
    val_ds   = PersonalColorDataset("data", split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = HierarchicalPersonalColorModel().to(device)

    criterion_s1 = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_s2 = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 1~3 에폭: 헤드만 학습 (백본 고정)
    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = optim.AdamW(
        list(model.stage1_head.parameters()) +
        list(model.stage2_heads.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # 4 에폭부터 전체 파인튜닝
        if epoch == 4:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY
            )
            logger.info("Switched to full fine-tuning (backbone unfrozen).")

        # ── 훈련 ──────────────────────────────────────────────────────
        model.train()
        total_loss, s1_correct, s2_correct, n = 0.0, 0, 0, 0

        for imgs, s_labels, t_labels in train_loader:
            imgs     = imgs.to(device)
            s_labels = s_labels.to(device)
            t_labels = t_labels.to(device)

            optimizer.zero_grad()
            s1_logits, s2_logits = model(imgs)  # (B,4), (B,4,3)

            loss_s1 = criterion_s1(s1_logits, s_labels)

            # Stage 2: 각 샘플의 실제 계절 헤드 손실
            s2_pred = torch.stack([
                s2_logits[i, s_labels[i], :]
                for i in range(imgs.size(0))
            ])                                         # (B, 3)
            loss_s2 = criterion_s2(s2_pred, t_labels)

            loss = loss_s1 + loss_s2
            loss.backward()
            optimizer.step()

            total_loss  += loss.item() * imgs.size(0)
            s1_correct  += (s1_logits.argmax(1) == s_labels).sum().item()
            s2_correct  += (s2_pred.argmax(1) == t_labels).sum().item()
            n           += imgs.size(0)

        scheduler.step()

        # ── 검증 ──────────────────────────────────────────────────────
        model.eval()
        vs1_c, vs2_c, vn = 0, 0, 0
        with torch.no_grad():
            for imgs, s_labels, t_labels in val_loader:
                imgs     = imgs.to(device)
                s_labels = s_labels.to(device)
                t_labels = t_labels.to(device)
                s1_l, s2_l = model(imgs)
                pred_s = s1_l.argmax(1)
                vs1_c += (pred_s == s_labels).sum().item()
                for i in range(imgs.size(0)):
                    pred_t = s2_l[i, s_labels[i], :].argmax()
                    vs2_c += int(pred_t == t_labels[i])
                vn += imgs.size(0)

        logger.info(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={total_loss/n:.4f} | "
            f"train s1={s1_correct/n:.3f} s2={s2_correct/n:.3f} | "
            f"val s1={vs1_c/vn:.3f} s2={vs2_c/vn:.3f}"
        )

        joint_acc = (vs1_c / vn) * (vs2_c / vn)
        if joint_acc > best_val_acc:
            best_val_acc = joint_acc
            torch.save(model.state_dict(), SAVE_PATH)
            logger.info(f"  ✓ Saved best model → {SAVE_PATH} (joint={joint_acc:.3f})")

    logger.info(f"Training complete. Best joint acc={best_val_acc:.3f}")


if __name__ == "__main__":
    train()
