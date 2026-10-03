#!/usr/bin/env python3
"""Optimizer and learning rate scheduler factory.

Builds the per-parameter weight-decay groups, instantiates the requested
optimizer (``adamw`` / ``sgd``), and pairs it with the chosen learning rate
scheduler (``onecycle``, ``cosine``, ``cyclic``, or none) plus an optional
linear warmup.
"""
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CyclicLR,
    OneCycleLR,
    SequentialLR,
    ConstantLR,
)


def configure_optimizers(
    named_parameters: dict = None,
    optimizer: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.001,
    betas: tuple = (0.9, 0.999),
    momentum: float = 0.9,
    scheduler: str = 'cosine',
    sch_warmpup_epochs: int = None,
    sch_warmpup_factor: float = 1,
    sch_T_0: int = 20,
    sch_T_mult: int = 2,
    sch_eta_min: float = 2e-6,
    total_steps: int = 0,
    steps_per_epoch: int = 0,
    **kwargs,
):
    """Build the optimizer and the learning rate schedule.

    LayerNorm weights and biases are placed in a no-decay parameter group
    while the remaining parameters use ``weight_decay``. The optimizer is
    paired with one of the supported schedulers and an optional linear
    warmup, returning a list compatible with PyTorch Lightning's
    :meth:`configure_optimizers` interface.

    Args:
        named_parameters: Iterable yielded by ``Module.named_parameters``.
        optimizer: Optimizer key, one of ``'adamw'`` or ``'sgd'``.
        learning_rate: Base learning rate.
        weight_decay: Weight decay applied to the decay group.
        betas: Beta coefficients for AdamW.
        momentum: Momentum for SGD.
        scheduler: Scheduler key: ``'onecycle'``, ``'cosine'``,
            ``'cyclic'``, or ``None`` / ``'none'`` for no scheduling.
        sch_warmpup_epochs: Linear warmup duration in epochs.
        sch_warmpup_factor: Multiplier applied during the warmup window.
        sch_T_0: ``T_0`` for :class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.
        sch_T_mult: ``T_mult`` for CosineAnnealingWarmRestarts.
        sch_eta_min: Minimum learning rate of the cosine and cyclic schedules.
        total_steps: Total number of training steps; required for
            ``'onecycle'``.
        steps_per_epoch: Steps per epoch; used by ``'cyclic'``.
        **kwargs: Ignored extra arguments (allows passing full train_config).

    Returns:
        Either ``[optimizer]`` or ``([optimizer], [scheduler])`` as
        expected by :class:`pytorch_lightning.LightningModule`.
    """
    no_decay = ['.norm', '.bias']
    params_decay: dict = {}
    params_nodecay: dict = {}

    for name, param in named_parameters:
        if not any(nd in name for nd in no_decay):
            params_decay[name] = param
        else:
            params_nodecay[name] = param

    optim_groups = [
        {'params': params_decay.values(), 'weight_decay': weight_decay},
        {'params': params_nodecay.values(), 'weight_decay': 0.0},
    ]

    if optimizer == 'adamw':
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
    elif optimizer == 'sgd':
        optimizer = SGD(optim_groups, lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose 'adamw' or 'sgd'.")

    warmup_scheduler = None
    if sch_warmpup_epochs is not None and sch_warmpup_epochs > 0:
        warmup_scheduler = ConstantLR(
            optimizer,
            factor=sch_warmpup_factor,
            total_iters=sch_warmpup_epochs,
        )

    if total_steps > 0 and scheduler == 'onecycle':
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
        )
    elif scheduler == 'cosine':
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sch_T_0,
            T_mult=sch_T_mult,
            eta_min=sch_eta_min,
        )
    elif scheduler == 'cyclic':
        lr_scheduler = CyclicLR(
            optimizer,
            base_lr=sch_eta_min,
            max_lr=learning_rate,
            step_size_up=steps_per_epoch // 2,
            step_size_down=steps_per_epoch // 2,
        )
    elif scheduler in ('none', None):
        lr_scheduler = None
    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler}'. "
            "Choose 'onecycle', 'cosine', 'cyclic', or None."
        )

    if warmup_scheduler is None and lr_scheduler is None:
        return [optimizer]
    if warmup_scheduler is not None and lr_scheduler is None:
        return [optimizer], [warmup_scheduler]
    if warmup_scheduler is None and lr_scheduler is not None:
        return [optimizer], [lr_scheduler]

    combined = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, lr_scheduler],
        milestones=[sch_warmpup_epochs],
    )
    return [optimizer], [combined]
