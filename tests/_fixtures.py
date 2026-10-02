"""Tiny synthetic data and model configs shared by the test suite.

Everything here is small enough to run on a laptop CPU in seconds.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from ageas.tool import Multimodal_Corpus

LOGREG_PARAMS = {
    'penalty': 'l2',
    'dual': False,
    'tol': 1e-4,
    'C': 1.0,
    'fit_intercept': True,
    'intercept_scaling': 1.0,
    'class_weight': None,
    'random_state': 0,
    'solver': 'lbfgs',
}

SVC_PARAMS = {
    'scaler_copy': True,
    'scaler_with_mean': True,
    'scaler_with_std': True,
    'C': 1.0,
    'kernel': 'linear',
    'degree': 3,
    'gamma': 'scale',
    'coef0': 0.0,
    'shrinking': True,
    'tol': 0.001,
    'cache_size': 200,
    'class_weight': None,
    'verbose': False,
    'max_iter': -1,
    'decision_func_shape': 'ovr',
    'break_ties': False,
    'random_state': 0,
}


MNB_PARAMS = {
    'alpha': 1.0,
    'force_alpha': True,
    'fit_prior': True,
    'class_prior': None,
}

XGB_PARAMS = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'multi:softprob',
    'max_depth': 2,
    'device': 'cpu',
    'seed': 0,
}

XGB_TRAIN_CONFIG = {
    'num_boost_round': 5,
    'early_stopping_rounds': None,
    'verbose_eval': False,
}


def write_configs(folder, units):
    """Write ``{model_type: {name: model_params}}`` as a hangar config folder."""
    for model_type, configs in units.items():
        type_dir = Path(folder) / model_type
        type_dir.mkdir(parents=True, exist_ok=True)
        for name, params in configs.items():
            with open(type_dir / f'{name}.json', 'w') as f:
                json.dump({'model_params': params}, f)


def corpus_from_matrix(x, labels):
    """In-memory labelled ``Multimodal_Corpus`` over a dense cell x gene array."""
    n_cells, n_genes = x.shape
    adata = ad.AnnData(
        csr_matrix(np.asarray(x, dtype=np.float32)),
        obs=pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)]),
        var=pd.DataFrame(index=[f'gene_{j}' for j in range(n_genes)]),
    )
    adata.obs['celltype'] = pd.Categorical(labels)
    adata.var['name'] = [f'name_{j}' for j in range(n_genes)]
    return Multimodal_Corpus(adata=adata, label_key='celltype', backed=False)


def marker_corpus(n_classes=3, n_per_class=5, n_noise_genes=3):
    """Corpus grouped by class, where gene ``k`` marks class ``k``.

    The last gene holds each cell's row number so tests can identify cells.
    """
    n_cells = n_classes * n_per_class
    y = np.repeat(np.arange(n_classes), n_per_class)
    x = np.zeros((n_cells, n_classes + n_noise_genes + 1), dtype=np.float32)
    x[np.arange(n_cells), y] = 1.0
    x[:, -1] = np.arange(n_cells)
    return corpus_from_matrix(x, [f'type_{k}' for k in y])


def marker_data(n_classes=2, n_per_class=15, n_genes=6, seed=0):
    """Non-negative data where gene ``k`` is a marker of class ``k``.

    Returns:
        ``(x, y, gene_names)``: ``x`` has the corpus batch shape
        ``(n_cells, 1, n_genes)``, ``y`` holds class ids ``0..n_classes-1``.
    """
    rng = np.random.RandomState(seed)
    n_cells = n_classes * n_per_class
    x = np.abs(rng.normal(0.0, 0.5, size=(n_cells, n_genes)))
    y = np.repeat(np.arange(n_classes), n_per_class)
    for k in range(n_classes):
        x[y == k, k] += 4.0
    order = rng.permutation(n_cells)
    x, y = x[order], y[order]
    gene_names = [f'g{i}' for i in range(n_genes)]
    return (
        torch.tensor(x, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y, dtype=torch.long),
        gene_names,
    )
