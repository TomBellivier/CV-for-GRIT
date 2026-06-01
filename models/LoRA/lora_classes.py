from lora_config import *

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from torchvision import transforms as T

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO

class LoRALinear(nn.Module):
    """adaptateur lora pour nn.Linear (W_base gelé, seuls A et B entraînés)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features  = base_layer.in_features
        out_features = base_layer.out_features
        self.rank    = rank
        self.scaling = alpha / rank

        self.lora_A  = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (f"rank={self.rank}, scaling={self.scaling:.3f}, "
                f"in={self.base_layer.in_features}, out={self.base_layer.out_features}")


class LoRAConv2d(nn.Module):
    """adaptateur lora pour nn.Conv2d via deux convolutions séquentielles de rang réduit."""

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_ch  = base_layer.in_channels
        out_ch = base_layer.out_channels
        k      = base_layer.kernel_size
        pad    = base_layer.padding
        stride = base_layer.stride

        self.rank    = rank
        self.scaling = alpha / rank

        self.lora_down = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.lora_up   = nn.Conv2d(rank, out_ch, k,
                                   padding=pad, stride=stride, bias=False)
        self.dropout   = nn.Dropout2d(dropout)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.scaling * self.lora_up(self.lora_down(self.dropout(x)))

    def extra_repr(self) -> str:
        return (f"rank={self.rank}, scaling={self.scaling:.3f}, "
                f"in_ch={self.base_layer.in_channels}, out_ch={self.base_layer.out_channels}")

class LoRAInjector:
    """remplace récursivement les couches cibles par leurs équivalents lora."""

    def __init__(self, config: LoRAConfig):
        self.config   = config
        self._injected: List[str] = []
        self._skipped:  List[str] = []

    def inject(self, model: nn.Module) -> nn.Module:
        """injecte lora dans model (in-place) et retourne le modèle."""
        self._replace_recursive(model, prefix="")
        return model

    def _should_replace(self, module: nn.Module, name: str) -> bool:
        module_type = type(module).__name__
        if module_type not in self.config.target_modules:
            return False
        if isinstance(module, nn.Linear):
            return min(module.in_features, module.out_features) >= self.config.min_weight_size
        if isinstance(module, nn.Conv2d):
            return min(module.in_channels, module.out_channels) >= self.config.min_weight_size
        return False

    def _replace_recursive(self, module: nn.Module, prefix: str):
        for child_name, child_module in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if self._should_replace(child_module, full_name):
                setattr(module, child_name, self._wrap(child_module))
                self._injected.append(full_name)
            else:
                self._replace_recursive(child_module, full_name)

    def _wrap(self, module: nn.Module) -> nn.Module:
        cfg = self.config
        if isinstance(module, nn.Linear):
            return LoRALinear(module, cfg.rank, cfg.alpha, cfg.dropout)
        if isinstance(module, nn.Conv2d):
            return LoRAConv2d(module, cfg.rank, cfg.alpha, cfg.dropout)
        raise TypeError(f"type non supporté : {type(module)}")

    @property
    def injected_layers(self) -> List[str]:
        return self._injected


def freeze_base_model(model: nn.Module) -> int:
    """gèle tous les paramètres sauf les adaptateurs lora."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ("lora_A", "lora_B", "lora_down", "lora_up")):
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """retourne total / trainable / frozen."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


class LoRAWeightManager:
    """
    sauvegarde, charge et échange les poids lora par groupe d'insectes.
    seuls lora_A / lora_B / lora_down / lora_up sont stockés → stockage minimal.
    """

    LORA_KEYS = ("lora_A", "lora_B", "lora_down.weight", "lora_up.weight")

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}

    @staticmethod
    def extract_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
        """extrait uniquement les paramètres lora du state_dict."""
        return {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if any(key in k for key in LoRAWeightManager.LORA_KEYS)
        }

    @staticmethod
    def apply_lora_state(model: nn.Module, lora_state: Dict[str, torch.Tensor]):
        """charge des poids lora dans le modèle sans modifier les poids de base."""
        current_state = model.state_dict()
        current_state.update(lora_state)
        model.load_state_dict(current_state, strict=False)

    def save_group(self, group_name: str, model: nn.Module, metadata: Optional[dict] = None):
        """sauvegarde les poids lora d'un groupe sur disque."""
        lora_state = self.extract_lora_state(model)
        checkpoint = {
            "group": group_name,
            "lora_state": lora_state,
            "metadata": metadata or {},
            "param_count": sum(v.numel() for v in lora_state.values()),
        }
        path = self.save_dir / f"lora_{group_name}.pt"
        torch.save(checkpoint, path)
        self._cache[group_name] = lora_state
        logger.info(f"  → groupe '{group_name}' sauvegardé : {path} "
                    f"({checkpoint['param_count']:,} params lora)")
        return path

    def load_group(self, group_name: str, model: nn.Module) -> dict:
        """charge et applique les poids lora d'un groupe."""
        if group_name in self._cache:
            self.apply_lora_state(model, self._cache[group_name])
            return {"source": "cache"}

        path = self.save_dir / f"lora_{group_name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"pas de checkpoint pour le groupe '{group_name}' : {path}")
        checkpoint = torch.load(path, map_location=DEVICE)
        self.apply_lora_state(model, checkpoint["lora_state"])
        self._cache[group_name] = checkpoint["lora_state"]
        logger.info(f"  ← groupe '{group_name}' chargé depuis {path}")
        return checkpoint.get("metadata", {})

    def summary(self):
        """affiche un tableau des groupes disponibles sur disque."""
        files = list(self.save_dir.glob("lora_*.pt"))
        if not files:
            print("aucun groupe sauvegardé.")
            return
        print(f"{'Groupe':<20} {'Paramètres LoRA':>18} {'Taille (Mo)':>12}")
        print("-" * 55)
        for f in sorted(files):
            ck = torch.load(f, map_location="cpu")
            size_mb = f.stat().st_size / 1e6
            print(f"{ck['group']:<20} {ck['param_count']:>18,} {size_mb:>12.3f}")

class YOLOPoseLoRA(nn.Module):
    """
    encapsule un modèle yolo-pose avec adaptateurs lora injectés.

    utilisation :
        model = YOLOPoseLoRA("yolov8n-pose.pt", lora_cfg)
        model.switch_group("coleoptera", lora_manager)
        preds = model(images)
    """

    def __init__(
        self,
        base_model_path: str,
        lora_config: LoRAConfig,
    ):
        super().__init__()
        yolo = YOLO(base_model_path)
        yolo = YOLO("config.yaml")
        
        self.backbone: nn.Module = yolo.model
        self.backbone.eval()

        injector = LoRAInjector(lora_config)
        injector.inject(self.backbone)
        freeze_base_model(self.backbone)

        stats = count_parameters(self.backbone)
        self._lora_config  = lora_config
        self._active_group: Optional[str] = None

        logger.info("YOLOPoseLoRA initialisé :")
        logger.info(f"  • paramètres totaux   : {stats['total']:>12,}")
        logger.info(f"  • paramètres gelés    : {stats['frozen']:>12,}")
        logger.info(f"  • paramètres lora     : {stats['trainable']:>12,}  "
                    f"({100*stats['trainable']/stats['total']:.2f}% du total)")
        logger.info(f"  • couches lora        : {len(injector.injected_layers)}")

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def switch_group(self, group_name: str, manager: LoRAWeightManager):
        """charge les poids lora du groupe spécifié."""
        manager.load_group(group_name, self.backbone)
        self._active_group = group_name
        logger.info(f"groupe actif → '{group_name}'")

    def save_current_group(self, manager: LoRAWeightManager, metadata: Optional[dict] = None):
        if self._active_group is None:
            raise ValueError("aucun groupe actif à sauvegarder.")
        return manager.save_group(self._active_group, self.backbone, metadata)

    def get_lora_params(self) -> List[nn.Parameter]:
        """retourne uniquement les paramètres lora pour l'optimiseur."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    @property
    def active_group(self) -> Optional[str]:
        return self._active_group


class InsectPoseDataset(Dataset):
    """
    dataset yolo-format pour la pose estimation d'insectes.

    structure attendue :
        data/<group>/images/train|val/
        data/<group>/labels/train|val/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 640,
        num_keypoints: int = 17,
        augment: bool = True,
    ):
        self.root      = Path(root_dir)
        self.img_size  = img_size
        self.num_kp    = num_keypoints
        self.augment   = augment and split == "train"
        self.img_dir   = self.root / "images" / split
        self.lbl_dir   = self.root / "labels" / split
        self.samples   = self._scan_samples()
        self.transform = self._build_transform()
        logger.info(f"dataset [{split}] : {len(self.samples)} images — {self.root.name}")

    def _scan_samples(self) -> List[Path]:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        imgs = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in exts])
        return [p for p in imgs if (self.lbl_dir / p.with_suffix(".txt").name).exists()]

    def _build_transform(self) -> T.Compose:
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.augment:
            return T.Compose([
                T.ToPILImage(),
                T.ColorJitter(0.3, 0.3, 0.2, 0.05),
                T.RandomHorizontalFlip(0.3),
                T.ToTensor(),
                normalize,
            ])
        return T.Compose([T.ToPILImage(), T.ToTensor(), normalize])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        lbl_path = self.lbl_dir / img_path.with_suffix(".txt").name

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = self.transform(img)

        targets = []
        with open(lbl_path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) >= 5 + 3 * self.num_kp:
                    targets.append(vals)

        targets_tensor = torch.zeros(len(targets), 5 + 3 * self.num_kp)
        for i, t in enumerate(targets):
            targets_tensor[i] = torch.tensor(t[:5 + 3 * self.num_kp])

        return img_tensor, targets_tensor, str(img_path)

def collate(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), labels, paths

def build_dataloaders(
    group_name: str,
    data_dir: str,
    cfg: TrainingConfig,
    insect_cfg_: InsectConfig,
) -> Tuple[DataLoader, DataLoader]:
    """crée les dataloaders train/val pour un groupe d'insectes."""

    train_ds = InsectPoseDataset(data_dir, "train", cfg.img_size, insect_cfg_.num_keypoints)
    val_ds   = InsectPoseDataset(data_dir, "val",   cfg.img_size, insect_cfg_.num_keypoints, augment=False)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate)

    return train_dl, val_dl
