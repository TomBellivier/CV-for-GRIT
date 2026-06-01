import logging
from typing import Dict, List
from dataclasses import dataclass, field

import numpy as np


import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {DEVICE}")
print(f"pytorch : {torch.__version__}")


@dataclass
class LoRAConfig:
    """hyperparamètres lora."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["Conv2d", "Linear"]
    )
    min_weight_size: int = 16


@dataclass
class TrainingConfig:
    """hyperparamètres d'entraînement."""
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    save_dir: str = "./checkpoints"
    img_size: int = 640


@dataclass
class InsectConfig:
    """configuration des groupes d'insectes."""
    groups: Dict[str, str] = field(default_factory=lambda: {
        #"lepidoptera": "../datasets/Lepidoptera",
        "hymenoptera": "../datasets/Hymenoptera",
        #"coleoptera": "../datasets/Coleoptera"
    })
    num_keypoints: int = 42
    base_model: str = "config.yaml"


lora_cfg = LoRAConfig()
train_cfg = TrainingConfig()
insect_cfg = InsectConfig()
