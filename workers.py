import logging, shutil, traceback
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image as PILImage
from PyQt6.QtCore import QThread, pyqtSignal
from constants import SEASON_CLASSES, SUPPORTED_EXT
from preprocessors import BasePreprocessor
from datasets import PersonalColorDataset


class DatasetLoaderWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    def __init__(self, folder_path):
        super().__init__(); self.folder_path = folder_path; self._running = True
    def run(self):
        try:    self.finished.emit(PersonalColorDataset(self.folder_path).get_class_distribution())
        except Exception as e: self.error.emit(str(e))
    def stop(self):
        self._running = False; self.quit(); self.wait(2000)


class PreprocessWorker(QThread):
    progress    = pyqtSignal(int, str)
    image_ready = pyqtSignal(np.ndarray, dict)
    finished    = pyqtSignal(list)
    error       = pyqtSignal(str)
    def __init__(self, paths, preprocessor):
        super().__init__(); self.paths = paths; self.preprocessor = preprocessor; self._running = True
    def run(self):
        all_meta = []; total = len(self.paths)
        for i, path in enumerate(self.paths):
            if not self._running: break
            try:
                img, meta = self.preprocessor.preprocess(path)
                all_meta.append(meta)
                if i == 0: self.image_ready.emit(img.copy(), meta)
                self.progress.emit(int((i+1)/total*100), Path(path).name)
            except Exception as e: logging.warning(f"전처리 실패 {path}: {e}")
        self.finished.emit(all_meta)
    def stop(self):
        self._running = False; self.quit(); self.wait(3000)


class TrainWorker(QThread):
    log_message    = pyqtSignal(str)
    epoch_complete = pyqtSignal(int, float, float, float, float)
    progress       = pyqtSignal(int)
    finished       = pyqtSignal(str)
    error          = pyqtSignal(str)
    def __init__(self, config):
        super().__init__(); self.config = config; self._running = True
    def run(self):
        try: self._train_loop()
        except Exception as e: self.error.emit(str(e)); self.log_message.emit(traceback.format_exc())
    def _train_loop(self):
        req = self.config.get("device","cpu")
        if req=="cuda" and not torch.cuda.is_available(): req="cpu"; self.log_message.emit("[WARN] CUDA unavailable -> CPU")
        if req=="mps"  and not torch.backends.mps.is_available(): req="cpu"; self.log_message.emit("[WARN] MPS unavailable -> CPU")
        device = torch.device(req); self.log_message.emit(f"[INFO] Device: {device}")
        _M=[0.485,0.456,0.406]; _S=[0.229,0.224,0.225]
        trn_tf = transforms.Compose([transforms.Resize((256,256)), transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(), transforms.Normalize(_M,_S)])
        val_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(_M,_S)])
        ds = PersonalColorDataset(self.config["data_dir"], transform=trn_tf)
        if len(ds)==0: self.error.emit("빈 데이터셋"); return
        nc = len(ds.classes); vs = max(1,int(0.2*len(ds))); ts = len(ds)-vs
        tr_sub, vl_sub = random_split(ds,[ts,vs], generator=torch.Generator().manual_seed(42))
        class _VW(Dataset):
            def __init__(s, sub, tf): s._sub=sub; s._tf=tf
            def __len__(s): return len(s._sub)
            def __getitem__(s, i):
                od=s._sub.dataset; oi=s._sub.indices[i]; p,l=od.samples[oi]
                try: img=PILImage.open(p).convert("RGB")
                except: img=PILImage.new("RGB",(224,224),(128,128,128))
                return s._tf(img), l
        bs = self.config.get("batch_size",16)
        tr_dl = DataLoader(tr_sub, batch_size=bs, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
        vl_dl = DataLoader(_VW(vl_sub,val_tf), batch_size=bs, num_workers=0)
        self.log_message.emit(f"[INFO] Classes:{ds.classes} Train:{ts} Val:{vs}")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        inf = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3,True), nn.Linear(inf,256), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256,nc))
        model = model.to(device)
        lr = self.config.get("lr",1e-3)
        opt = optim.AdamW([{"params":model.features.parameters(),"lr":lr*0.1},{"params":model.classifier.parameters(),"lr":lr}], weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config.get("epochs",10))
        crit = nn.CrossEntropyLoss(); best=0.; sp=self.config.get("save_path","best_model.pt"); ep_n=self.config.get("epochs",10)
        for ep in range(1, ep_n+1):
            if not self._running: self.log_message.emit("[INFO] 중단"); break
            model.train(); tl=tc=tt=0
            for bi,(x,y) in enumerate(tr_dl):
                if not self._running: break
                x,y=x.to(device),y.to(device); opt.zero_grad(); o=model(x); lo=crit(o,y); lo.backward(); opt.step()
                tl+=lo.item()*x.size(0); tc+=o.argmax(1).eq(y).sum().item(); tt+=x.size(0)
                self.progress.emit(int((bi+1)/len(tr_dl)*100))
            model.eval(); vl=vc=vt=0
            with torch.no_grad():
                for x,y in vl_dl:
                    x,y=x.to(device),y.to(device); o=model(x); lo=crit(o,y)
                    vl+=lo.item()*x.size(0); vc+=o.argmax(1).eq(y).sum().item(); vt+=x.size(0)
            atl=tl/max(tt,1); avl=vl/max(vt,1); ta=tc/max(tt,1)*100; va=vc/max(vt,1)*100
            sch.step()
            self.log_message.emit(f"[Epoch {ep:03d}/{ep_n}] Train Loss={atl:.4f} Acc={ta:.2f}% | Val Loss={avl:.4f} Acc={va:.2f}%")
            self.epoch_complete.emit(ep, atl, ta, avl, va)
            if va>best:
                best=va; torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"val_acc":va,"classes":ds.classes,"num_classes":nc},sp)
                self.log_message.emit(f"  [BEST] val_acc={va:.2f}% -> {sp}")
        self.log_message.emit(f"\n{'='*60}\n[DONE] Best Val Acc: {best:.2f}%\n{'='*60}"); self.finished.emit(sp)
    def stop(self): self._running = False


class InferenceWorker(QThread):
    finished = pyqtSignal(str, list)
    error    = pyqtSignal(str)
    def __init__(self, model_path, image_path):
        super().__init__(); self.model_path=model_path; self.image_path=image_path
    def run(self):
        try: self.finished.emit(*self._infer())
        except Exception as e: self.error.emit(str(e))
    def _infer(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(self.model_path, map_location=dev); nc=ck.get("num_classes",4); cls=ck.get("classes",SEASON_CLASSES[:nc])
        m = models.efficientnet_b0(weights=None); inf=m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3,True), nn.Linear(inf,256), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256,nc))
        m.load_state_dict(ck["model_state_dict"]); m=m.to(dev).eval()
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        t = tf(PILImage.open(self.image_path).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad(): pr = torch.softmax(m(t),1).squeeze().cpu().numpy()
        pc = cls[int(pr.argmax())]; rk = sorted(zip(cls,pr.tolist()), key=lambda x:x[1], reverse=True)
        return pc, rk


class AutoLabelWorker(QThread):
    progress    = pyqtSignal(int, str, str, float)
    log_message = pyqtSignal(str)
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)
    def __init__(self, model_path, src_dir, out_dir, threshold=0.85):
        super().__init__(); self.model_path=model_path; self.src_dir=src_dir
        self.out_dir=out_dir; self.threshold=threshold; self._running=True
    def run(self):
        try: self._auto_label()
        except Exception as e: self.error.emit(str(e)); self.log_message.emit(traceback.format_exc())
    def _auto_label(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.log_message.emit(f"[INFO] Device: {dev}")
        ck = torch.load(self.model_path, map_location=dev); nc=ck.get("num_classes",4); cls=ck.get("classes",SEASON_CLASSES[:nc])
        m = models.efficientnet_b0(weights=None); inf=m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3,True), nn.Linear(inf,256), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256,nc))
        m.load_state_dict(ck["model_state_dict"]); m=m.to(dev).eval()
        self.log_message.emit(f"[INFO] Model loaded  classes={cls}")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        imgs = [p for p in Path(self.src_dir).rglob("*") if p.suffix.lower() in SUPPORTED_EXT]
        total = len(imgs); self.log_message.emit(f"[INFO] Images:{total}  thr:{self.threshold:.0%}")
        if total==0: self.finished.emit({"error":"이미지 없음"}); return
        out = Path(self.out_dir)
        for c in cls: (out/c).mkdir(parents=True, exist_ok=True)
        stats: Dict = {c:0 for c in cls}; stats["skipped_low_conf"]=0; stats["skipped_error"]=0
        for i, ip in enumerate(imgs):
            if not self._running: self.log_message.emit("[INFO] 중단"); break
            try:
                t = tf(PILImage.open(str(ip)).convert("RGB")).unsqueeze(0).to(dev)
                with torch.no_grad(): pr = torch.softmax(m(t),1).squeeze().cpu().numpy()
                cf=float(pr.max()); pc=cls[int(pr.argmax())]; pct=int((i+1)/total*100)
                self.progress.emit(pct, ip.name, pc, cf)
                if cf>=self.threshold:
                    dst=out/pc/ip.name
                    if dst.exists(): dst=out/pc/f"{ip.stem}_{i}{ip.suffix}"
                    shutil.copy2(str(ip), str(dst)); stats[pc]+=1
                    self.log_message.emit(f"  [OK]  [{pc:6s}] {ip.name}  conf={cf:.3f}")
                else:
                    stats["skipped_low_conf"]+=1
                    self.log_message.emit(f"  [SKIP] {ip.name}  conf={cf:.3f}<{self.threshold:.2f}  (best={pc})")
            except Exception as e: stats["skipped_error"]+=1; self.log_message.emit(f"  [ERR] {ip.name}: {e}")
        lb = sum(stats[c] for c in cls)
        self.log_message.emit(f"\n{'='*50}\n[DONE] 분류:{lb}  건너뜀:{stats['skipped_low_conf']}  오류:{stats['skipped_error']}\n{'='*50}")
        self.finished.emit(stats)
    def stop(self): self._running = False
