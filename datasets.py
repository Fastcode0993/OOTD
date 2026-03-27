from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage
from constants import SEASON_CLASSES, SUPPORTED_EXT


class PersonalColorDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples:      List[Tuple[str,int]] = []
        self.classes:      List[str]            = []
        self.class_to_idx: Dict[str,int]        = {}
        self._scan()

    def _scan(self):
        found    = sorted(d.name for d in self.root_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        priority = {c:i for i,c in enumerate(SEASON_CLASSES)}
        self.classes      = sorted(found, key=lambda x: priority.get(x,999))
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        for cls in self.classes:
            for p in (self.root_dir/cls).iterdir():
                if p.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((str(p), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:    img = PILImage.open(path).convert("RGB")
        except: img = PILImage.new("RGB",(224,224),(128,128,128))
        if self.transform: img = self.transform(img)
        return img, label

    def get_class_distribution(self) -> Dict[str,int]:
        dist = {c:0 for c in self.classes}
        for _, lbl in self.samples: dist[self.classes[lbl]] += 1
        return dist
