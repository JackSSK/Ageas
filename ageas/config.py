#!/usr/bin/env python3
"""Allowed keys of a unit config.

A unit config is the JSON a :class:`~ageas.Hangar` loads for one unit::

    {"max_epochs": ..., "model_params": {...}, "train_config": {...}}

Every key must be one the code reads; an unknown key is an error rather
than being silently ignored, so a misspelt or stale key can't quietly change
what a unit trains with.
"""
from ageas.classical.log_reg import DEFAULT_PARAMS as LOGREG_DEFAULTS
from ageas.classical.mnb import DEFAULT_PARAMS as MNB_DEFAULTS
from ageas.classical.svc import DEFAULT_PARAMS as SVC_DEFAULTS

#: Keys allowed at the top level of a unit config (they reach the trainer).
TOP_LEVEL_KEYS = frozenset({
    'max_epochs', 'model_params', 'train_config', 'pretrained_ckpt',
    'default_root_dir', 'enable_progress_bar', 'dataloader_workers',
})

#: Lightning ``Trainer`` settings accepted inside an NN ``train_config``.
TRAINER_KEYS = frozenset({
    'precision', 'num_nodes', 'log_every_n_steps', 'accumulate_grad_batches',
    'gradient_clip_val', 'gradient_clip_algorithm', 'ckpt_every_n_epochs',
    'save_last', 'monitor', 'save_top_k_ckpt', 'gradient_accum_schedule',
    'enable_checkpointing',
})

#: Loss, optimizer and scheduler settings of an NN ``train_config``.
OPTIMIZER_KEYS = frozenset({
    'loss_reduction', 'optimizer', 'learning_rate', 'weight_decay', 'betas',
    'momentum', 'scheduler', 'sch_warmpup_epochs', 'sch_warmpup_factor',
    'sch_T_0', 'sch_T_mult', 'sch_eta_min', 'total_steps', 'steps_per_epoch',
})

#: Filled in by the trainer from the data; allowed but not needed in configs.
_DERIVED_NN_KEYS = frozenset({'num_classes', 'len_in', 'inplanes'})

_NN_MODEL_KEYS = frozenset({
    'block', 'block_nums', 'block_dims', 'latent_fea_dim', 'dropout',
    'norm_layer', 'bias', 'block_layer_type', 'block_num_layer',
    'nonlinearity', 'bidirectional', 'proj_size',
}) | _DERIVED_NN_KEYS

_MIXER_MODEL_KEYS = frozenset({
    'block', 'block_nums', 'block_planes', 'latent_planes', 'latent_fea_dim',
    'global_conv', 'dropout', 'zero_init_residual', 'groups', 'dilation',
    'width_per_group', 'replace_stride_with_dilation', 'norm_layer',
}) | _DERIVED_NN_KEYS

#: ``model_params`` keys per unit type. XGBoost parameters are left to
#: XGBoost, which reports parameters it doesn't use.
MODEL_KEYS = {
    'mlp': _NN_MODEL_KEYS,
    'rnn': _NN_MODEL_KEYS,
    'resnet': _MIXER_MODEL_KEYS,
    'logreg': frozenset(LOGREG_DEFAULTS) | {'num_class'},
    'svc': frozenset(SVC_DEFAULTS) | {'num_class'},
    'mnb': frozenset(MNB_DEFAULTS) | {'num_class'},
    'xgb': None,
}

_NN_TRAIN_KEYS = frozenset({'batch_size'}) | TRAINER_KEYS | OPTIMIZER_KEYS

#: ``train_config`` keys per unit type. For XGBoost they are the keyword
#: arguments of :func:`xgboost.train`.
TRAIN_KEYS = {
    'mlp': _NN_TRAIN_KEYS,
    'rnn': _NN_TRAIN_KEYS,
    'resnet': _NN_TRAIN_KEYS,
    'logreg': frozenset({'batch_size'}),
    'svc': frozenset({'batch_size'}),
    'mnb': frozenset({'batch_size'}),
    'xgb': frozenset({
        'batch_size', 'num_boost_round', 'evals', 'obj', 'maximize',
        'early_stopping_rounds', 'evals_result', 'verbose_eval', 'xgb_model',
        'callbacks', 'custom_metric',
    }),
}


def validate_unit_config(unit_type: str, config: dict, source: str = 'config') -> None:
    """Raise if ``config`` holds a key the code would not read.

    Unit types Ageas doesn't support are not checked here; the unit warns
    about them when it is launched.

    Args:
        unit_type: Model type key, e.g. ``'mlp'`` or ``'logreg'``.
        config: The unit config dict.
        source: Where the config came from (e.g. its file path), for the
            error message.

    Raises:
        ValueError: Naming every unknown key and ``source``.
    """
    if unit_type not in MODEL_KEYS:
        return

    unknown = [key for key in config if key not in TOP_LEVEL_KEYS]
    model_keys = MODEL_KEYS[unit_type]
    if model_keys is not None:
        unknown += [
            f'model_params.{key}'
            for key in config.get('model_params') or {}
            if key not in model_keys
        ]
    unknown += [
        f'train_config.{key}'
        for key in config.get('train_config') or {}
        if key not in TRAIN_KEYS[unit_type]
    ]
    if unknown:
        raise ValueError(
            f"Unknown key(s) {', '.join(repr(k) for k in unknown)} in "
            f"{unit_type} config {source}."
        )
