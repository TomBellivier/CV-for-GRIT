"""
Low-rank adapters (LoRA) for the convolutions of a YOLO-pose model.

Principle
---------
A convolution weight ``W`` of shape [C_out, C_in, k, k] is kept frozen, and a
low-rank correction is learned in parallel:

    y = conv(x, W) + (alpha / r) * conv(conv(x, A), B)

with ``A`` of shape [r, C_in, k, k] and ``B`` of shape [C_out, r, 1, 1].
``B`` is initialised to zero, so the adapted model starts out numerically
identical to the base model. The trainable parameter count is
``r * (C_in * k * k + C_out)`` instead of ``C_out * C_in * k * k``.

One adapter bank is held per insect group. Switching groups is a single string
assignment -- the shared weights are never reloaded, which is the entire point:
one backbone in memory, four behaviours.

Usage
-----
    model = YOLO("base.pt")
    inject_lora(model.model, groups=["Coleoptera"], rank=8, targets="neck_head")
    set_active_group(model.model, "Coleoptera")
    freeze_base(model.model)
"""

import math
import re

import torch
import torch.nn as nn

LORA_ATTR = "lora_adapters"      # ModuleDict attribute name on each wrapper
LORA_KEY = "lora_adapters."      # substring used to filter parameters


class LoRAConv2d(nn.Module):
    """Wrap a frozen Conv2d with one low-rank adapter per group."""

    def __init__(self, base: nn.Conv2d, groups, rank=8, alpha=None):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.active_group = None

        in_ch = base.in_channels
        out_ch = base.out_channels
        # A rank above min(C_in, C_out) buys nothing but parameters.
        self.rank = max(1, min(self.rank, in_ch, out_ch))
        self.scale = self.alpha / self.rank

        adapters = {}
        for name in groups:
            down = nn.Conv2d(in_ch, self.rank, base.kernel_size,
                             stride=base.stride, padding=base.padding,
                             dilation=base.dilation, groups=1, bias=False)
            up = nn.Conv2d(self.rank, out_ch, 1, bias=False)
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            nn.init.zeros_(up.weight)
            adapters[_safe_key(name)] = nn.Sequential(down, up)
        setattr(self, LORA_ATTR, nn.ModuleDict(adapters))

    def forward(self, x):
        out = self.base(x)
        if self.active_group is None:
            return out
        bank = getattr(self, LORA_ATTR)
        # nn.ModuleDict has no .get(); membership must be tested explicitly.
        if self.active_group not in bank:
            return out
        return out + self.scale * bank[self.active_group](x)

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, active={self.active_group}"


def _safe_key(name):
    """ModuleDict keys cannot contain dots."""
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def _layer_index(module_name):
    """Top-level index inside the PoseModel Sequential, or None."""
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def resolve_targets(spec, n_layers=24):
    """Turn a target spec into a set of top-level layer indices.

    'all'        -> every layer
    'neck_head'  -> 11..23 (top-down FPN, bottom-up PAN and the head)
    'head'       -> 23 only
    '13,16,19'   -> explicit list
    """
    spec = str(spec).strip().lower()
    if spec == "all":
        return set(range(n_layers))
    if spec == "neck_head":
        return set(range(11, n_layers))
    if spec == "neck":
        return set(range(11, 23))
    if spec == "head":
        return {n_layers - 1}
    return {int(tok) for tok in re.split(r"[,\s]+", spec) if tok}


def inject_lora(pose_model, groups, rank=8, alpha=None, targets="neck_head",
                skip_grouped=True, min_channels=16, verbose=True):
    """Replace Conv2d layers by LoRAConv2d in the selected top-level layers.

    Parameters
    ----------
    pose_model : the torch module held in ``YOLO(...).model``
    groups     : list of group names, one adapter bank each
    skip_grouped : leave depthwise convolutions (groups != 1) untouched. A
        low-rank factorisation of a depthwise kernel is close to meaningless,
        and the dense ``A`` it would require is disproportionately expensive.
    min_channels : skip very narrow convolutions, where a rank-r adapter would
        hold more parameters than the layer it adapts.
    """
    n_layers = len(pose_model.model) if hasattr(pose_model, "model") else 24
    wanted = resolve_targets(targets, n_layers)

    replacements = []
    for name, module in pose_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        index = _layer_index(name)
        if index is None or index not in wanted:
            continue
        if skip_grouped and module.groups != 1:
            continue
        if min(module.in_channels, module.out_channels) < min_channels:
            continue
        replacements.append((name, module))

    n_params = 0
    for name, module in replacements:
        parent_name, _, attr = name.rpartition(".")
        parent = pose_model.get_submodule(parent_name) if parent_name else pose_model
        wrapper = LoRAConv2d(module, groups, rank=rank, alpha=alpha)
        wrapper.to(module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr, wrapper)
        n_params += sum(p.numel() for p in getattr(wrapper, LORA_ATTR).parameters())

    if verbose:
        total = sum(p.numel() for p in pose_model.parameters())
        print(f"[lora] wrapped {len(replacements)} conv layers "
              f"(targets={targets}, rank={rank})")
        print(f"[lora] adapter parameters: {n_params:,} for {len(groups)} group(s) "
              f"-> {n_params / max(len(groups), 1):,.0f} per group "
              f"({100 * n_params / max(total, 1):.2f}% of the full model)")
    return pose_model


def set_active_group(pose_model, group_name):
    """Activate one adapter bank (or None to fall back to the base model)."""
    key = _safe_key(group_name) if group_name is not None else None
    for module in pose_model.modules():
        if isinstance(module, LoRAConv2d):
            module.active_group = key
    return pose_model


def freeze_base(pose_model, also_train=(), verbose=True):
    """Freeze everything except the LoRA adapters.

    ``also_train`` accepts substrings matched against parameter names, e.g.
    ``("model.23.",)`` to leave the detection/pose head trainable as well.
    """
    n_train, n_frozen = 0, 0
    for name, param in pose_model.named_parameters():
        trainable = LORA_KEY in name or any(tok in name for tok in also_train)
        param.requires_grad = trainable
        if trainable:
            n_train += param.numel()
        else:
            n_frozen += param.numel()
    if verbose:
        total = n_train + n_frozen
        print(f"[lora] trainable {n_train:,} / {total:,} parameters "
              f"({100 * n_train / max(total, 1):.2f}%)")
    return pose_model


def lora_state_dict(pose_model):
    """Extract only the adapter tensors."""
    return {k: v.detach().cpu().clone()
            for k, v in pose_model.state_dict().items() if LORA_KEY in k}


def load_lora_state_dict(pose_model, state, group_name=None, strict=False):
    """Load adapter tensors, optionally remapping them onto another group key.

    Training runs one group at a time, so each checkpoint holds a single bank.
    ``group_name`` lets the evaluation script drop that bank into the right slot
    of a four-group model.
    """
    if group_name is not None:
        target = _safe_key(group_name)
        remapped = {}
        for key, value in state.items():
            head, _, tail = key.partition(LORA_KEY)
            bank_key, _, rest = tail.partition(".")
            remapped[f"{head}{LORA_KEY}{target}.{rest}"] = value
        state = remapped

    missing, unexpected = pose_model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if LORA_KEY in k]
    if strict and unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected[:5]}")
    loaded = len(state) - len(unexpected)
    print(f"[lora] loaded {loaded}/{len(state)} adapter tensors"
          + (f" into bank '{group_name}'" if group_name else ""))
    return pose_model


def extract_lora_from_checkpoint(ckpt_path):
    """Pull the adapter tensors out of an Ultralytics checkpoint.

    ``best.pt`` pickles the whole nn.Module, so ``lora.py`` must be importable
    when this runs -- which it is, since this function lives in it.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    module = ckpt.get("ema") or ckpt.get("model")
    if module is None:
        raise RuntimeError(f"no model found inside {ckpt_path}")
    state = module.float().state_dict()
    return {k: v.detach().cpu().clone() for k, v in state.items() if LORA_KEY in k}
