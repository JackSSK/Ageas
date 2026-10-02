#!/usr/bin/env python3
"""Utility toolkit for Ageas.

Re-exports the dataset corpora (:class:`Multimodal_Corpus`,
:class:`Repr_Corpus`), the k-fold splitter, the optimizer factory, the
generic Integrated-Gradients explainer, the trainer factory used by the
deck, and a handful of scaling, normalization, and ``AnnData`` helpers.
"""
import time

import anndata as ad
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from .config_optim import configure_optimizers
from .corpus_loader import Multimodal_Corpus, Repr_Corpus, kfold_random_split
from .explainer import Basic_Clf_Explainer as Basic_Clf_Explainer
from .knn import KMeans, KNN
from .trainer import Trainer_Maker


def z_score_scaler(x, mode: str = 'numpy', safe_division: float = 1.0):
    """Z-score standardize a tensor or array.

    Args:
        x: Input data (:class:`torch.Tensor` or :class:`numpy.ndarray`).
        mode: Backend to use, one of ``'numpy'`` or ``'torch'``.
        safe_division: Replacement standard deviation used when ``std == 0``
            to avoid division by zero.

    Returns:
        Standardized tensor/array of the same type as ``x``.

    Raises:
        ValueError: If ``mode`` is not ``'numpy'`` or ``'torch'``.
    """
    if mode == 'torch':
        mean = torch.mean(x)
        std = torch.std(x)
    elif mode == 'numpy':
        mean = np.mean(x)
        std = np.std(x)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'numpy' or 'torch'.")
    if std == 0:
        std = safe_division
    return (x - mean) / std


def min_max_scaler(x, mode: str = 'numpy'):
    """Min-max scale ``x`` into the ``[0, 1]`` range.

    Args:
        x: Input data (:class:`torch.Tensor` or :class:`numpy.ndarray`).
        mode: Backend to use, one of ``'numpy'`` or ``'torch'``.

    Returns:
        Scaled values in ``[0, 1]``, or an all-zeros array if
        ``min == max``.

    Raises:
        ValueError: If ``mode`` is not ``'numpy'`` or ``'torch'``.
    """
    if mode == 'torch':
        minimum = torch.min(x)
        maximum = torch.max(x)
    elif mode == 'numpy':
        minimum = np.min(x)
        maximum = np.max(x)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'numpy' or 'torch'.")

    if minimum == maximum:
        return np.full_like(x, 0.0)
    return (x - minimum) / (maximum - minimum)


def l1_normalize(tensor, mode: str = 'torch'):
    """L1-normalize a tensor or array.

    Returns ``tensor`` unchanged if the L1 norm is zero.

    Args:
        tensor: Input values (:class:`torch.Tensor` or
            :class:`numpy.ndarray`).
        mode: Backend to use, one of ``'torch'`` or ``'numpy'``.

    Returns:
        ``tensor`` divided by its L1 norm, or the unmodified ``tensor`` if
        the L1 norm is zero.

    Raises:
        ValueError: If ``mode`` is not ``'torch'`` or ``'numpy'``.
    """
    if mode == 'torch':
        abs_sum = torch.sum(torch.abs(tensor))
        return tensor if abs_sum == 0 else tensor / abs_sum
    if mode == 'numpy':
        abs_sum = np.sum(np.abs(tensor))
        return tensor if abs_sum == 0 else tensor / abs_sum
    raise ValueError(f"Unsupported mode '{mode}'. Choose 'torch' or 'numpy'.")


def get_time_stamp() -> str:
    """Return the current local time as a ``YYYY-MM-DD HH:MM:SS`` string."""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def make_fake_adata(
    n_cells: int = 100,
    n_genes: int = 20,
    n_informative: int = 2,
    n_redundant: int = 2,
    n_repeated: int = 0,
    n_class: int = 2,
    n_clusters_per_class: int = 2,
    seed: int = 42,
) -> ad.AnnData:
    """Generate a synthetic ``AnnData`` for tests and tutorials.

    Wraps :func:`sklearn.datasets.make_classification` to produce a
    classification problem and packages it as an ``AnnData`` with named
    informative, redundant, and repeated genes plus a categorical
    ``celltype`` column in ``adata.obs``.

    Args:
        n_cells: Number of cells (samples).
        n_genes: Total number of genes (features).
        n_informative: Number of class-informative features.
        n_redundant: Number of redundant features built from the informative
            ones.
        n_repeated: Number of repeated (duplicated) features.
        n_class: Number of class labels.
        n_clusters_per_class: Clusters per class for
            :func:`~sklearn.datasets.make_classification`.
        seed: Random seed forwarded to ``seed_everything`` and
            ``make_classification``.

    Returns:
        Sparse :class:`anndata.AnnData` object with shape
        ``(n_cells, n_genes)``.
    """
    seed_everything(seed)

    # Unshuffled, make_classification orders columns informative, redundant,
    # repeated, then noise; shuffling would permute columns away from these
    # names. Rows are shuffled separately below.
    X, y = make_classification(
        n_samples=n_cells,
        n_features=n_genes,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_class,
        n_clusters_per_class=n_clusters_per_class,
        shuffle=False,
        random_state=seed,
    )
    row_order = np.random.RandomState(seed).permutation(n_cells)
    X, y = X[row_order], y[row_order]
    standard_genes = n_genes - n_informative - n_redundant - n_repeated

    var_index = (
        [f'informative_gene_{i}' for i in range(n_informative)]
        + [f'redundant_gene_{i}' for i in range(n_redundant)]
        + [f'repeated_gene_{i}' for i in range(n_repeated)]
        + [f'fake_gene_{i}' for i in range(standard_genes)]
    )

    adata = ad.AnnData(
        csr_matrix(X),
        obs=pd.DataFrame(index=[f'fake_cell_{i}' for i in range(n_cells)]),
        var=pd.DataFrame(index=var_index),
    )
    adata.obs['celltype'] = pd.Categorical(y, categories=range(n_class))
    adata.var['name'] = adata.var.index
    return adata


__all__ = [
    'Multimodal_Corpus',
    'Repr_Corpus',
    'Basic_Clf_Explainer',
    'Trainer_Maker',
    'kfold_random_split',
    'configure_optimizers',
    'z_score_scaler',
    'min_max_scaler',
    'l1_normalize',
    'get_time_stamp',
    'make_fake_adata',
    'KNN',
    'KMeans',
]
