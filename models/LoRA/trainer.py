from ultralytics.utils import IterableSimpleNamespace
from custom_yolo import PoseLoss26_v2

from lora_config import *
from loss import *
from lora_classes import *

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class LoRATrainer:
    """
    entraîne les adaptateurs lora pour un groupe d'insectes.
    le backbone yolo est gelé ; seuls les poids a et b sont mis à jour.
    """

    def __init__(
        self,
        model: YOLOPoseLoRA,
        manager: LoRAWeightManager,
        config: TrainingConfig,
    ):
        self.model   = model
        self.manager = manager
        self.cfg     = config
        self.history: Dict[str, Dict[str, List[float]]] = {}

    def _build_optimizer(self):
        return AdamW(self.model.get_lora_params(),
                     lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _warmup_lr(self, optimizer, epoch: int):
        if epoch < self.cfg.warmup_epochs:
            factor = (epoch + 1) / self.cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = self.cfg.lr * factor

    def _train_epoch(self, loader: DataLoader, optimizer, epoch: int) -> float:
        self.model.backbone.train()
        # maintenir bn en mode eval (poids gelés)
        for m in self.model.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                m.eval()

        total_loss = 0.0
        for imgs, targets_list, _ in loader:
            imgs = imgs.to(DEVICE)
            assert not imgs.isnan().any(),    "NaN dans les images"
            assert not imgs.isinf().any(),    "Inf dans les images"
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                preds = self.model(imgs)

                if isinstance(preds, dict):
                    preds_p = preds["one2many"]
                elif isinstance(preds, (list, tuple)):
                    preds_p = preds[1]["one2many"]
                else:
                    raise TypeError(f"format de prédiction inattendu : {type(preds)}")

                # Vérifier les preds avant loss
                if preds_p["kpts"].isnan().any():
                    for k, v in preds.items():
                        if isinstance(v, torch.Tensor):
                            print(f"{k}: nan={v.isnan().any().item()} inf={v.isinf().any().item()} "
                                f"min={v.min().item():.3f} max={v.max().item():.3f}")
                loss  = self._compute_loss(preds, targets_list)
                loss  = loss + lora_regularization(self.model.backbone)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.get_lora_params(), self.cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.backbone.eval()
        total_loss = 0.0
        for imgs, targets_list, _ in loader:
            imgs  = imgs.to(DEVICE)
            preds = self.model(imgs)
            loss  = self._compute_loss(preds, targets_list)
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _ensure_hyp_namespace(self):
        """
        normalise self.model.backbone.model[-1].loss.hyp vers IterableSimpleNamespace.
        selon la version d'ultralytics, hyp peut être un dict, un SimpleNamespace,
        ou déjà un IterableSimpleNamespace — les deux premiers cas provoquent un
        TypeError lors de l'appel à la loss.
        """
        head_loss = self.model.backbone.model[-1].loss
        if not hasattr(head_loss, "hyp"):
            return
        hyp = head_loss.hyp
        if isinstance(hyp, IterableSimpleNamespace):
            return
        if isinstance(hyp, dict):
            head_loss.hyp = IterableSimpleNamespace(**hyp)
        else:
            # SimpleNamespace ou tout autre objet avec des attributs
            head_loss.hyp = IterableSimpleNamespace(**vars(hyp))

    def _compute_loss(self, preds, targets_list) -> torch.Tensor:
        """assemble le batch et appelle la loss interne du head yolo."""

        batch_idx, classes, bboxes, keypoints = [], [], [], []
        num_kp = insect_cfg.num_keypoints

        for i, t in enumerate(targets_list):
            if t.shape[0] == 0:
                continue
            t = t.to(DEVICE)
            n = t.shape[0]
            batch_idx.append(torch.full((n,), i, dtype=torch.float32, device=DEVICE))
            classes.append(t[:, 0])
            bboxes.append(t[:, 1:5])
            keypoints.append(t[:, 5:5 + 3 * num_kp].view(n, num_kp, 3))

        if not batch_idx:
            return torch.tensor(0.0, device=DEVICE, requires_grad=True)

        batch = {
            "batch_idx": torch.cat(batch_idx),
            "cls":       torch.cat(classes),
            "bboxes":    torch.cat(bboxes),
            "keypoints": torch.cat(keypoints),
            "img":       torch.zeros(len(targets_list), device=DEVICE),
        }

        if isinstance(preds, dict):
            preds = preds["one2many"]
        elif isinstance(preds, (list, tuple)):
            preds = preds[1]["one2many"]
        else:
            raise TypeError(f"format de prédiction inattendu : {type(preds)}")
    
        self.model.backbone.args = {}
        self.model.backbone.args["box"] = 7.5
        self.model.backbone.args["cls"] = 0.5
        self.model.backbone.args["dfl"] = 1.5
        self.model.backbone.args["pose"] = 12.0
        self.model.backbone.args["kobj"] = 1.0

        loss_fn = PoseLoss26_v2(self.model.backbone)

        loss_res = preds["boxes"].isnan().any()

        print(loss_res)

        loss, _ = loss_fn(preds, batch)

        return loss.sum()

    def train_group(
        self,
        group_name: str,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        resume: bool = False,
    ) -> Dict[str, List[float]]:
        """entraîne les adaptateurs lora pour group_name."""
        logger.info(f"\n{'='*60}")
        logger.info(f"entraînement groupe : {group_name.upper()}")
        logger.info(f"{'='*60}")

        if resume:
            try:
                self.manager.load_group(group_name, self.model.backbone)
                logger.info("reprise depuis checkpoint existant.")
            except FileNotFoundError:
                logger.info("pas de checkpoint — initialisation à zéro.")
        else:
            self._reset_lora_weights()

        self.model._active_group = group_name

        optimizer = self._build_optimizer()
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs - self.cfg.warmup_epochs,
            eta_min=self.cfg.lr * 0.01,
        )

        train_losses, val_losses = [], []
        best_val = float("inf")

        for epoch in range(self.cfg.epochs):
            self._warmup_lr(optimizer, epoch)
            train_loss = self._train_epoch(train_loader, optimizer, epoch)
            val_loss   = self._val_epoch(val_loader)

            if epoch >= self.cfg.warmup_epochs:
                scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                self.manager.save_group(
                    group_name, self.model.backbone,
                    metadata={"epoch": epoch, "val_loss": val_loss}
                )

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  epoch {epoch+1:>3}/{self.cfg.epochs} "
                    f"| train={train_loss:.4f} | val={val_loss:.4f} "
                    f"| lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        history = {"train": train_losses, "val": val_losses}
        self.history[group_name] = history
        logger.info(f"✓ groupe '{group_name}' terminé. best val loss : {best_val:.4f}")
        return history

    def _reset_lora_weights(self):
        """réinitialise a ~ kaiming, b = 0."""
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, LoRALinear):
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)
            elif isinstance(module, LoRAConv2d):
                nn.init.kaiming_uniform_(module.lora_down.weight, a=math.sqrt(5))
                nn.init.zeros_(module.lora_up.weight)


def train_all_groups(
    model: YOLOPoseLoRA,
    trainer: LoRATrainer,
    insect_config: InsectConfig,
    train_config: TrainingConfig,
    dry_run: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """entraîne séquentiellement un adaptateur lora par groupe. dry_run=True simule sans données."""
    all_histories = {}

    for group_name, data_dir in insect_config.groups.items():
        logger.info(f"\npréparation du groupe : {group_name}")

        if dry_run or not Path(data_dir).exists():
            logger.warning(f"  [dry run] dossier '{data_dir}' absent — simulation.")
            all_histories[group_name] = {
                "train": [float('nan')] * train_config.epochs,
                "val":   [float('nan')] * train_config.epochs,
            }
            continue

        train_dl, val_dl = build_dataloaders(
            group_name, data_dir, train_config, insect_config
        )

        if len(train_dl.dataset) == 0:
            logger.warning(f"  dataset vide pour '{group_name}', ignoré.")
            continue

        history = trainer.train_group(
            group_name   = group_name,
            train_loader = train_dl,
            val_loader   = val_dl,
            resume       = False,
        )
        all_histories[group_name] = history

    return all_histories

def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    figsize: Tuple[int, int] = (15, 8),
):
    """affiche les courbes de loss train/val pour chaque groupe."""
    groups = [g for g, h in histories.items() if not all(math.isnan(v) for v in h["train"])]
    if not groups:
        print("pas de données réelles (dry_run) — affichage simulé.")
        np.random.seed(42)
        groups = list(histories.keys())
        for g in groups:
            n = train_cfg.epochs
            histories[g]["train"] = (0.5 * np.exp(-np.linspace(0, 3, n)) + 0.05 + 0.01*np.random.randn(n)).tolist()
            histories[g]["val"]   = (0.6 * np.exp(-np.linspace(0, 2.5, n)) + 0.08 + 0.015*np.random.randn(n)).tolist()

    n_groups = len(groups)
    cols = min(3, n_groups)
    rows = math.ceil(n_groups / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, n_groups))

    for i, (group, color) in enumerate(zip(groups, colors)):
        ax = axes[i]
        h  = histories[group]
        epochs = range(1, len(h["train"]) + 1)
        ax.plot(epochs, h["train"], color=color, lw=2, label="train")
        ax.plot(epochs, h["val"],   color=color, lw=2, ls="--", label="val", alpha=0.7)
        ax.set_title(group.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(h["train"]))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("courbes d'entraînement lora par groupe d'insectes",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()