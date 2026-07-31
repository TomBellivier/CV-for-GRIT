"""
Group-specific BatchNorm ("domain-specific BN") for a YOLO-pose model.

Principle
---------
Every convolution weight is shared across the four insect groups. The only
thing that is duplicated is the normalisation layer: its affine parameters
(gamma, beta) and its running statistics (mean, var). On this architecture that
is well under 1% of the parameters, which is why it is the only specialisation
scheme that 2 000 images can actually support.

The intuition: a large part of the gap between two visually distinct domains is
a per-channel shift and rescaling of the feature distributions. Re-estimating
the normalisation absorbs it without touching the filters themselves.

Training does not need this module: it is enough to fine-tune a copy of the
base model with everything except the BatchNorm layers frozen, then keep the
resulting BN tensors. ``GroupBatchNorm2d`` exists so that the four results can
be reassembled into a *single* deployable model at evaluation time.
"""

import copy

import torch
import torch.nn as nn

BN_KEYS = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


class GroupBatchNorm2d(nn.Module):
    """A bank of BatchNorm2d layers, one per group, sharing everything else."""

    def __init__(self, base: nn.BatchNorm2d, groups):
        super().__init__()
        self.group_names = list(groups)
        self.banks = nn.ModuleList(
            [copy.deepcopy(base) for _ in self.group_names])
        self.active_index = 0

    @property
    def active(self):
        return self.banks[self.active_index]

    def set_group(self, group_name):
        if group_name in self.group_names:
            self.active_index = self.group_names.index(group_name)
        return self

    def forward(self, x):
        return self.banks[self.active_index](x)

    def extra_repr(self):
        name = self.group_names[self.active_index] if self.group_names else "-"
        return f"groups={len(self.group_names)}, active='{name}'"


def convert_to_group_bn(pose_model, groups, verbose=True):
    """Replace every BatchNorm2d by a GroupBatchNorm2d bank."""
    targets = [(name, module) for name, module in pose_model.named_modules()
               if isinstance(module, nn.BatchNorm2d)]

    n_params = 0
    for name, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = pose_model.get_submodule(parent_name) if parent_name else pose_model
        bank = GroupBatchNorm2d(module, groups)
        bank.to(module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr, bank)
        n_params += sum(p.numel() for p in bank.parameters())

    if verbose:
        total = sum(p.numel() for p in pose_model.parameters())
        print(f"[group-bn] converted {len(targets)} BatchNorm layers into "
              f"{len(groups)}-way banks")
        print(f"[group-bn] group-specific parameters: {n_params:,} total "
              f"-> {n_params // max(len(groups), 1):,} per group "
              f"({100 * n_params / max(total, 1) / max(len(groups), 1):.2f}% "
              f"of the model each)")
    return pose_model


def set_active_group(pose_model, group_name):
    """Point every bank at one group."""
    for module in pose_model.modules():
        if isinstance(module, GroupBatchNorm2d):
            module.set_group(group_name)
    return pose_model


def freeze_except_bn(pose_model, also_train=(), verbose=True):
    """Freeze every parameter except BatchNorm affine terms.

    Running statistics are buffers, not parameters: they keep updating on every
    forward pass in training mode regardless of ``requires_grad``. That is
    intended -- re-estimating them on the group's own images is half of what
    this method does.
    """
    bn_param_names = set()
    for name, module in pose_model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, GroupBatchNorm2d)):
            for param_name, _ in module.named_parameters():
                bn_param_names.add(f"{name}.{param_name}")

    n_train, n_frozen = 0, 0
    for name, param in pose_model.named_parameters():
        trainable = name in bn_param_names or \
            any(tok in name for tok in also_train)
        param.requires_grad = trainable
        if trainable:
            n_train += param.numel()
        else:
            n_frozen += param.numel()

    if verbose:
        total = n_train + n_frozen
        print(f"[group-bn] trainable {n_train:,} / {total:,} parameters "
              f"({100 * n_train / max(total, 1):.2f}%)")
    return pose_model


def bn_state_dict(pose_model):
    """Extract the tensors of every plain BatchNorm2d, keyed by module path."""
    state = {}
    for name, module in pose_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for key, value in module.state_dict().items():
                state[f"{name}.{key}"] = value.detach().cpu().clone()
    return state


def load_group_bn_state(pose_model, state, group_name, strict=True):
    """Install a plain-BN state dict into one bank of a converted model.

    ``state`` is keyed on the original module paths (``model.13.cv1.bn.weight``);
    the bank version needs ``model.13.cv1.bn.banks.<i>.weight``.
    """
    index = None
    for module in pose_model.modules():
        if isinstance(module, GroupBatchNorm2d):
            if group_name not in module.group_names:
                raise KeyError(f"group '{group_name}' is not in this model "
                               f"({module.group_names})")
            index = module.group_names.index(group_name)
            break
    if index is None:
        raise RuntimeError("model has no GroupBatchNorm2d layers; "
                           "call convert_to_group_bn first")

    remapped = {}
    for key, value in state.items():
        module_path, _, leaf = key.rpartition(".")
        remapped[f"{module_path}.banks.{index}.{leaf}"] = value

    missing, unexpected = pose_model.load_state_dict(remapped, strict=False)
    if strict and unexpected:
        raise RuntimeError(
            f"{len(unexpected)} BatchNorm tensors did not fit the model "
            f"(first: {unexpected[0]}). The base weights probably differ from "
            f"the ones used during group fine-tuning.")
    print(f"[group-bn] bank '{group_name}' loaded "
          f"({len(remapped) - len(unexpected)}/{len(remapped)} tensors)")
    return pose_model


def extract_bn_from_checkpoint(ckpt_path):
    """Pull BatchNorm tensors out of an Ultralytics checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    module = ckpt.get("ema") or ckpt.get("model")
    if module is None:
        raise RuntimeError(f"no model found inside {ckpt_path}")
    return bn_state_dict(module.float())


def check_shared_weights(ckpt_paths, atol=1e-6):
    """Verify that the non-BN weights really did stay identical across groups.

    Worth running once: if this fails, the freezing did not take effect and the
    approach silently degenerated into four independent full models.
    """
    reference = None
    for path in ckpt_paths:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        module = (ckpt.get("ema") or ckpt.get("model")).float()
        bn_names = {name for name, mod in module.named_modules()
                    if isinstance(mod, nn.BatchNorm2d)}
        state = {k: v for k, v in module.state_dict().items()
                 if k.rsplit(".", 1)[0] not in bn_names}
        if reference is None:
            reference = (path, state)
            continue
        for key, value in state.items():
            other = reference[1].get(key)
            if other is None or other.shape != value.shape:
                return False, f"{key}: shape mismatch vs {reference[0]}"
            if not torch.allclose(other.float(), value.float(), atol=atol):
                delta = (other.float() - value.float()).abs().max().item()
                return False, f"{key}: max |delta| = {delta:.3e} vs {reference[0]}"
    return True, "shared weights are identical across all groups"
