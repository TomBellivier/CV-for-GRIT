"""
Thin compatibility layer over Ultralytics' internals.

Ultralytics renames private helpers between minor releases. Anything this
codebase borrows from ``ultralytics.utils`` goes through here, so a rename costs
one edit instead of a hunt through every training script.

Known history for the model-unwrapping helper:

* up to ~8.3      ``de_parallel(model)``  -- strips DataParallel / DDP
* from ~8.4       ``unwrap_model(model)`` -- also strips torch.compile
                                             (``._orig_mod``)
"""


def unwrap_model(model):
    """Return the underlying module, stripped of compile/parallel wrappers."""
    try:
        from ultralytics.utils.torch_utils import unwrap_model as _impl
        return _impl(model)
    except ImportError:
        pass
    try:
        from ultralytics.utils.torch_utils import de_parallel as _impl
        return _impl(model)
    except ImportError:
        pass

    # Last resort: replicate the behaviour without Ultralytics.
    import torch.nn as nn
    while True:
        inner = getattr(model, "_orig_mod", None)
        if isinstance(inner, nn.Module):
            model = inner
            continue
        inner = getattr(model, "module", None)
        if isinstance(inner, nn.Module):
            model = inner
            continue
        return model


def disable_fuse(pose_model):
    """Make ``model.fuse()`` a no-op on this instance.

    Ultralytics fuses Conv+BatchNorm before every ``predict`` and ``val`` call,
    and the operation is **destructive and in-place**. That is fine for a plain
    model, but fatal for both group-specialisation schemes:

    * LoRA -- ``fuse_conv_and_bn`` writes into ``conv.weight.data`` and would
      have to be told about the adapter branch;
    * group BatchNorm -- the fused module would bake in *one* group's statistics
      and the other three banks would become unreachable.

    Skipping fusion costs a little inference speed and nothing else. Correctness
    of the group switching is worth far more here than a few percent of latency.
    """
    pose_model.fuse = lambda *args, **kwargs: pose_model
    return pose_model