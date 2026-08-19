"""Diagnostic du gel LoRA : ou les parametres redeviennent-ils entrainables ?

A lancer depuis la racine du projet :
    python diagnostic_lora.py

Le script instrumente le trainer a chaque etape et affiche le ratio de parametres
entrainables. La premiere ligne ou le ratio remonte designe le coupable.
"""

from __future__ import annotations

import re
from typing import Any

from insectpose.cli import load_config
from insectpose.registry import load_all_plugins

load_all_plugins()


def report(model: Any, label: str) -> None:
    """Ratio de parametres entrainables, et echantillon de ce qui l'est encore."""
    total = trainable = 0
    noms: list[str] = []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            noms.append(name)
    ratio = trainable / max(total, 1)
    print(f"  {label:38s} {ratio:6.1%}  ({trainable:>11,} / {total:,})")
    if ratio > 0.5:
        hors_lora = [n for n in noms if "lora_" not in n][:4]
        print(f"      entrainables hors LoRA : {hors_lora}")


def main() -> None:
    from ultralytics import YOLO

    from insectpose.approaches.lora import LoraApproach
    from insectpose.training.patching import head_index, pose_trainer_class

    cfg = load_config(["approach=lora", "train.device=cpu"])
    approach = LoraApproach(cfg)

    print("\n=== 1. Modele nu ===")
    model = YOLO(str(cfg.approach.weights))
    report(model.model, "avant tout patch")

    print("\n=== 2. Apres injection peft ===")
    approach._apply_lora(model.model)
    report(model.model, "juste apres inject_adapter_in_model")

    print("\n=== 3. Apres notre _freeze ===")
    approach._freeze(model.model)
    report(model.model, "apres LoraApproach._freeze")

    names = [n for n, _ in model.model.named_modules()]
    last = head_index(names)
    print(f"\n  index de tete detecte : model.{last}")
    print(f"  cibles LoRA           : {len(getattr(approach, '_lora_targets', []))}")
    print(f"  depthwise ecartees    : {len(getattr(approach, '_lora_skipped', []))}")
    if getattr(approach, "_lora_targets", None):
        print(f"  exemples              : {approach._lora_targets[:3]}")

    print("\n=== 4. Ce que le trainer verrait ===")
    # `self.model` du trainer peut etre enveloppe (torch.compile, DDP) : le gel
    # s'appliquerait alors a une enveloppe, et les noms de parametres seraient prefixes.
    trainer_cls = pose_trainer_class()
    print(f"  classe de trainer     : {trainer_cls.__name__}")
    prefixes = {n.split(".")[0] for n, _ in model.model.named_parameters()}
    print(f"  prefixes de parametres: {sorted(prefixes)[:5]}")

    print("\n=== 5. Motifs de gel appliques ===")
    trainable_patterns = [r"lora_[AB]"]
    if bool(cfg.approach.lora.train_head):
        trainable_patterns.append(rf"^model\.{last}\.")
    print(f"  motifs 'a garder'     : {trainable_patterns}")

    compiled = [re.compile(p) for p in trainable_patterns]
    gardes = [n for n, _ in model.model.named_parameters()
              if any(c.search(n) for c in compiled)]
    total_p = sum(p.numel() for _, p in model.model.named_parameters())
    gardes_p = sum(p.numel() for n, p in model.model.named_parameters()
                   if any(c.search(n) for c in compiled))
    print(f"  parametres correspondant aux motifs : {len(gardes)} tenseurs, "
          f"{gardes_p:,} / {total_p:,} = {gardes_p / total_p:.1%}")
    print("\n  => si ce dernier ratio vaut deja ~67%, le probleme est le MOTIF")
    print("     (la tete pese trop), pas le mecanisme de gel.")

    tete = sum(p.numel() for n, p in model.model.named_parameters()
               if n.startswith(f"model.{last}."))
    lora = sum(p.numel() for n, p in model.model.named_parameters() if "lora_" in n)
    print(f"\n  poids de la tete model.{last}. : {tete:,} ({tete / total_p:.1%})")
    print(f"  poids des adaptateurs        : {lora:,} ({lora / total_p:.1%})")


if __name__ == "__main__":
    main()