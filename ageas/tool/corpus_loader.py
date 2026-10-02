#!/usr/bin/env python3
"""Dataset corpora and split utilities used by Ageas.

Defines the in-memory :class:`Tensor_Corpus` that wraps tensor data, the
file-backed :class:`Multimodal_Corpus` and :class:`Repr_Corpus` built on top
of ``AnnData``, and the ``kfold_random_split`` helper used by the selection
ops.
"""
import logging
from collections import Counter

import anndata as ad
import hdf5plugin  # noqa: F401 — registers HDF5 filters
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

_logger = logging.getLogger(__name__)


def decomp_batch(batch) -> tuple:
    """Split a dataloader batch into ``(x, y)``.

    Tolerates batches that are tensors, single-element tuples, or
    ``(x, y)`` tuples; returns ``y = None`` whenever no labels are available.

    Args:
        batch: A tensor, a length-1 tuple/list, or a ``(x, y)`` pair.

    Returns:
        Tuple ``(x, y)`` where ``y`` may be ``None``.
    """
    if isinstance(batch, (list, tuple)):
        assert len(batch) <= 2
        if len(batch) == 2:
            return batch[0], batch[1]
        return batch[0], None
    return batch, None


def get_all_data(data: Dataset) -> tuple:
    """Materialise an entire dataset into a single ``(x, y)`` tuple.

    Constructs a single-batch :class:`~torch.utils.data.DataLoader` so that
    one pass yields all samples; useful for converting the deck's per-fold
    subsets into in-memory :class:`Tensor_Corpus` instances.

    Args:
        data: Any :class:`~torch.utils.data.Dataset` instance.

    Returns:
        Tuple ``(x, y)`` with all samples concatenated.
    """
    loader = DataLoader(data, batch_size=len(data), shuffle=False)
    batch = next(iter(loader))
    return decomp_batch(batch)


class Tensor_Corpus(Dataset):
    """In-memory ``Dataset`` wrapping pre-materialised tensor data.

    Used by the splitting helpers to convert ``AnnData``-backed corpora into
    RAM-resident copies, while still propagating the parent's ``label_dict``
    and ``features`` for downstream stratification and explanation.

    Attributes:
        data: Feature tensor.
        labels: Label tensor (may be ``None``).
        parent: Original corpus this was derived from.
        label_key: Column key used for labels (inherited from parent).
        label_dict: Integer-to-label mapping (inherited from parent).
        features: Feature names (inherited from parent).
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor = None,
        parent: Dataset = None,
    ) -> None:
        """Initialize a Tensor_Corpus.

        Args:
            data: Feature tensor.
            labels: Optional label tensor.
            parent: Parent corpus to inherit metadata from.
        """
        self.data = data
        self.labels = labels
        self.parent = parent
        self.label_key = parent.label_key if parent is not None else None
        self.label_dict = parent.label_dict if parent is not None else None
        self.features = parent.features if parent is not None else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.data[idx]
        return self.data[idx], self.labels[idx]

    def get_class_distribution(self, translate_label: bool = True) -> dict:
        """Return the per-class sample count, sorted by frequency.

        Args:
            translate_label: If ``True`` and a parent ``label_dict`` is
                available, keys are translated from numeric IDs to their
                human-readable labels.

        Returns:
            ``{label: count}`` ordered most to least frequent.
        """
        class_counts = {
            k: v
            for k, v in sorted(
                Counter(self.labels.numpy()).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        if translate_label and self.parent is not None:
            class_counts = {
                self.parent.label_dict[k]: v
                for k, v in class_counts.items()
            }
        return class_counts

    def stratify(self, class_label: int = None) -> 'Subset':
        """Return a ``Subset`` containing only samples of ``class_label``.

        Args:
            class_label: Numeric class label to keep.

        Returns:
            A :class:`~torch.utils.data.Subset` filtered to the requested
            class, or ``self`` if no labels are available.
        """
        if self.labels is None:
            _logger.warning("Cannot stratify a dataset without labels.")
            return self
        indices = [
            i for i in range(len(self.labels))
            if self.labels[i].item() == class_label
        ]
        return Subset(self, indices)


class Multimodal_Corpus(Dataset):
    """``AnnData``-backed multimodal sample corpus.

    Each item is a stack of one or more layers from the underlying
    ``AnnData`` (``X`` plus any layer key). Optionally the modalities can be
    squeezed into a single feature axis for use with 1D classifiers.

    Attributes:
        monomodal: If set, only this layer name is returned per item.
        squeeze_multimodal: If ``True``, stacked layers are reshaped to
            ``(1, -1, 1)``.
        layers: Layer names to stack per item.
        label_key: ``adata.obs`` column used as categorical labels.
        dtype: Numeric dtype for returned data arrays.
        adata_path: Path to the backing ``.h5ad`` file (may be ``None``).
        adata_backed: Whether the file was opened in backed mode.
        adata: The underlying :class:`anndata.AnnData` object.
        label_dict: Integer-to-label mapping (``None`` if unlabelled).
        reverse_label_dict: Label-to-integer mapping.
    """

    def __init__(
        self,
        adata_path: str = None,
        adata: ad.AnnData = None,
        backed: bool = True,
        monomodal: str = None,
        layers: list = None,
        label_key: str = None,
        squeeze_multimodal: bool = False,
        dtype=np.float32,
    ) -> None:
        """Initialize a Multimodal_Corpus.

        Args:
            adata_path: Path to an ``.h5ad`` file. Ignored if ``adata`` is
                passed in directly.
            adata: In-memory ``AnnData`` object to use as the data source.
            backed: If ``True``, ``read_h5ad`` opens the file in backed mode.
            monomodal: If set, only the given layer name is returned per item.
            layers: Layer names to stack per item. ``'X'`` means ``adata.X``.
                Defaults to ``['X']``.
            label_key: ``adata.obs`` column to use as the categorical label.
                ``None`` produces an unlabelled corpus.
            squeeze_multimodal: If ``True`` (and ``monomodal`` is unset),
                reshape the stacked layers into a flat ``(1, -1, 1)`` tensor.
            dtype: Numeric dtype for the returned data array.
        """
        if layers is None:
            layers = ['X']

        self.monomodal = monomodal
        self.squeeze_multimodal = squeeze_multimodal
        self.layers = layers
        self.label_key = label_key
        self.dtype = dtype
        self.adata_path = adata_path
        self.adata_backed = backed

        self.adata = adata if adata is not None else ad.read_h5ad(adata_path, backed=backed)

        self.label_dict = (
            {
                i: label
                for i, label in enumerate(
                    self.adata.obs[label_key].cat.categories
                )
            }
            if self.label_key is not None
            else None
        )
        self.reverse_label_dict = (
            {v: k for k, v in self.label_dict.items()}
            if self.label_key is not None
            else None
        )

    @property
    def features(self) -> list:
        """List of feature (gene) names from ``adata.var.index``."""
        return self.adata.var.index.tolist()

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx):
        idx_data = self.adata[idx].to_memory()
        data_parts = []
        for layer in self.layers:
            if self.monomodal is not None and self.monomodal != layer:
                continue
            if layer == 'X':
                data_parts.append(idx_data.X.toarray())
            else:
                data_parts.append(idx_data.layers[layer].toarray())
        data = np.concatenate(data_parts, axis=0, dtype=self.dtype)

        if self.squeeze_multimodal and self.monomodal is None:
            data = np.reshape(data, (1, -1, 1))

        if self.label_key is None:
            return data
        label = self.reverse_label_dict[
            self.adata.obs[self.label_key].iloc[idx]
        ]
        return data, label

    def stratify(
        self,
        class_label: int = None,
        actual_label: str = None,
        label_key: str = None,
    ) -> 'Multimodal_Corpus':
        """Return a copy of the corpus filtered to a single class.

        At least one of ``class_label`` or ``actual_label`` must be provided.
        If both are provided they must agree under the parent's ``label_dict``.

        Args:
            class_label: Numeric class label.
            actual_label: Human-readable label string.
            label_key: ``adata.obs`` column to interpret as labels. Defaults
                to the corpus' own ``self.label_key``.

        Returns:
            A new :class:`Multimodal_Corpus` instance containing only the
            matching cells.

        Raises:
            ValueError: If neither ``class_label`` nor ``actual_label`` is
                provided, or if they are inconsistent.
        """
        if self.label_key is None:
            if label_key is None:
                _logger.warning("Cannot stratify a dataset without labels.")
                return self
            self.label_key = label_key
        assert label_key is None or label_key == self.label_key, \
            "Inconsistent label_key"

        if actual_label is None:
            if class_label is not None:
                actual_label = self.label_dict[class_label]
            else:
                raise ValueError("Invalid Dataset Stratification: E0")
        elif class_label is not None:
            assert actual_label == self.label_dict[class_label], \
                "Inconsistent class_label and actual_label"

        strat_ds = self.__class__(
            adata_path=self.adata_path,
            adata=self.adata[
                self.adata.obs[self.label_key] == actual_label
            ].copy(),
            monomodal=self.monomodal,
            layers=self.layers,
            label_key=self.label_key,
            squeeze_multimodal=self.squeeze_multimodal,
            backed=self.adata_backed,
            dtype=self.dtype,
        )
        # Keep the parent's class ids: built from the subset alone, the
        # mapping would renumber the single remaining class to 0.
        if self.label_dict:
            strat_ds.label_dict = self.label_dict.copy()
            strat_ds.reverse_label_dict = self.reverse_label_dict.copy()
        return strat_ds

    def copy(self) -> 'Multimodal_Corpus':
        """Return a deep copy of the dataset instance.

        Copies the underlying ``AnnData`` along with the ``label_dict`` /
        ``reverse_label_dict`` mappings so that destructive feature pruning
        in selection ops does not leak across copies.

        Returns:
            An independent :class:`Multimodal_Corpus` instance.
        """
        new_ds = self.__class__(
            adata_path=self.adata_path,
            adata=self.adata.copy(),
            backed=self.adata_backed,
            monomodal=self.monomodal,
            layers=self.layers,
            label_key=self.label_key,
            squeeze_multimodal=self.squeeze_multimodal,
            dtype=self.dtype,
        )
        if self.label_dict:
            new_ds.label_dict = self.label_dict.copy()
        if self.reverse_label_dict:
            new_ds.reverse_label_dict = self.reverse_label_dict.copy()
        return new_ds

    def restrict_features(self, features) -> None:
        """Keep only ``features`` (in the given order), in place.

        Args:
            features: Feature IDs from ``adata.var.index`` (a list, an index,
                or a boolean mask over the current features).
        """
        self.adata = self.adata[:, features]


class Repr_Corpus(Multimodal_Corpus):
    """``AnnData``-backed corpus that yields a precomputed representation.

    Items are taken from ``adata.obsm[rep_key]`` (e.g. ``'X_pca'``) instead
    of the gene-level layers, which is convenient when chaining Ageas after
    an upstream embedding model.

    Attributes:
        rep_key: Key in ``adata.obsm`` to use as the feature matrix.
    """

    def __init__(
        self,
        adata_path: str = None,
        adata: ad.AnnData = None,
        rep_key: str = 'X_pca',
        label_key: str = None,
        dtype=np.float32,
    ) -> None:
        """Initialize a Repr_Corpus.

        Args:
            adata_path: Path to an ``.h5ad`` file (ignored if ``adata``
                is provided).
            adata: In-memory ``AnnData`` object.
            rep_key: Key in ``adata.obsm`` used as the feature matrix.
            label_key: ``adata.obs`` column to use as categorical labels.
            dtype: Numeric dtype for returned data arrays.
        """
        self.adata_path = adata_path
        self.rep_key = rep_key
        self.label_key = label_key
        self.dtype = dtype

        self.adata = adata if adata is not None else ad.read_h5ad(adata_path, backed=False)

        self.label_dict = (
            {
                i: label
                for i, label in enumerate(
                    self.adata.obs[label_key].cat.categories
                )
            }
            if self.label_key is not None
            else None
        )
        self.reverse_label_dict = (
            {v: k for k, v in self.label_dict.items()}
            if self.label_key is not None
            else None
        )

    def __getitem__(self, idx):
        data = self.adata[idx].to_memory().obsm[self.rep_key]
        if self.label_key is None:
            return data, None
        label = self.reverse_label_dict[
            self.adata.obs[self.label_key].iloc[idx]
        ]
        return data, label


def _get_data_labels(dataset: Dataset, query_idx: list = None) -> list:
    """Read numeric labels for ``query_idx`` (or every item if ``None``).

    Args:
        dataset: An ``AnnData``-backed corpus.
        query_idx: Indices to query. ``None`` queries all observations.

    Returns:
        List of integer class labels.
    """
    if query_idx is None:
        query_idx = range(len(dataset.adata.obs.index))
    return [
        dataset.reverse_label_dict[
            dataset.adata.obs[dataset.label_key].iloc[idx]
        ]
        for idx in query_idx
    ]


def _can_stratify(labels: list, test_fraction: float) -> bool:
    """Whether ``train_test_split`` can stratify ``labels`` at this fraction.

    It needs every class to have at least two members and the held-out part
    to hold at least one cell per class.
    """
    counts = Counter(labels)
    n_test = int(np.ceil(test_fraction * len(labels)))
    return min(counts.values()) >= 2 and n_test >= len(counts)


def kfold_random_split(
    dataset,
    n_splits: int = 1,
    valid_fraction: float = 0.1,
    stratified_test: bool = False,
    stratified_valid: bool = True,
    oversample_method: str = None,
    oversample_by: str = 'median',
    random_seed: int = None,
) -> tuple:
    """K-fold split of an ``AnnData``-backed corpus.

    Args:
        dataset: Source corpus to split.
        n_splits: Number of folds.
        valid_fraction: Fraction of each training fold reserved for
            validation. ``None`` or ``0`` skips the validation split.
        stratified_test: If ``True`` use
            :class:`~sklearn.model_selection.StratifiedKFold`.
        stratified_valid: If ``True``, the validation split inside each
            fold is class-stratified.
        oversample_method: Optional oversampling strategy (``'repeat'``,
            ``'SMOTE'``, ``'ADASYN'``).
        oversample_by: Target class size when oversampling: ``'mean'``,
            ``'median'``, or ``'max'``.
        random_seed: Seed for reproducible splits.

    Returns:
        Tuple ``(train_list, valid_list, test_list)`` of length ``n_splits``.
    """
    train_list: list = []
    valid_list: list = []
    test_list: list = []

    shuffle = random_seed is not None
    if stratified_test:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_seed,
        ).split(dataset.adata.obs.index, dataset.adata.obs[dataset.label_key])
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_seed,
        ).split(dataset.adata.obs.index)

    for train_indices, test_indices in splitter:
        test_list.append(
            Tensor_Corpus(*get_all_data(Subset(dataset, test_indices)), parent=dataset)
        )

        if valid_fraction is not None and valid_fraction > 0:
            stratify = (
                _get_data_labels(dataset, train_indices) if stratified_valid else None
            )
            if stratify is not None and not _can_stratify(stratify, valid_fraction):
                _logger.warning(
                    "Validation split of %d cells cannot be stratified over "
                    "%d classes; using an unstratified split.",
                    int(np.ceil(valid_fraction * len(stratify))),
                    len(set(stratify)),
                )
                stratify = None
            train_indices, valid_indices = train_test_split(
                train_indices,
                test_size=valid_fraction,
                stratify=stratify,
                random_state=random_seed,
            )
            valid_list.append(
                Tensor_Corpus(
                    *get_all_data(Subset(dataset, valid_indices)),
                    parent=dataset,
                )
            )
        else:
            valid_list.append(None)

        train_data = Subset(dataset, train_indices)
        if oversample_method is None:
            train_list.append(
                Tensor_Corpus(*get_all_data(train_data), parent=dataset)
            )
        else:
            train_list.append(
                _oversample(
                    query=train_data,
                    parent=dataset,
                    oversample_method=oversample_method,
                    oversample_by=oversample_by,
                    random_seed=random_seed,
                )
            )

    return train_list, valid_list, test_list


def oversample_corpus(
    dataset,
    oversample_method: str,
    oversample_by: str = 'median',
    random_seed: int = None,
) -> Tensor_Corpus:
    """Oversample a whole corpus so all classes reach a target size.

    The same oversampling the k-fold training splits receive, applied to
    every cell of ``dataset``. See :func:`_oversample` for the methods.
    """
    return _oversample(
        query=Subset(dataset, list(range(len(dataset)))),
        parent=dataset,
        oversample_method=oversample_method,
        oversample_by=oversample_by,
        random_seed=random_seed,
    )


def _oversample(
    query: Dataset,
    parent: Dataset = None,
    oversample_method: str = None,
    oversample_by: str = 'median',
    random_seed: int = None,
) -> Tensor_Corpus:
    """Oversample a corpus so all classes reach a target size.

    The target class size is the mean / median / max of the per-class
    counts. Only the ``'repeat'`` strategy is currently active.

    Args:
        query: Training subset to oversample.
        parent: Parent corpus for label lookup. Defaults to ``query``.
        oversample_method: Oversampling strategy. Currently only
            ``'repeat'`` is supported.
        oversample_by: Target class size: ``'mean'``, ``'median'``, or
            ``'max'``.
        random_seed: Random seed for reproducible sampling.

    Returns:
        Oversampled :class:`Tensor_Corpus`.

    Raises:
        ValueError: If ``oversample_by`` is not recognised.
        DeprecationWarning: If ``'SMOTE'`` or ``'ADASYN'`` are requested.
    """
    parent = query if parent is None else parent

    class_counts = parent.adata[query.indices].obs[parent.label_key].value_counts()

    target_map = {
        'median': int(class_counts.median()),
        'mean': int(class_counts.mean()),
        'max': int(class_counts.max()),
    }
    if oversample_by.lower() not in target_map:
        raise ValueError(
            f"Invalid oversample_by '{oversample_by}'. "
            "Choose 'median', 'mean', or 'max'."
        )
    target_class_count = target_map[oversample_by.lower()]

    if oversample_method.lower() == 'repeat':
        rng = np.random.RandomState(random_seed)
        train_indices: list = []
        for label in parent.label_dict.values():
            label_indices = [
                x for x in query.indices
                if parent.adata.obs[parent.label_key].iloc[x] == label
            ]
            train_indices += label_indices
            # Top up a minority class with repeats; every real cell stays.
            n_missing = target_class_count - len(label_indices)
            if n_missing > 0:
                train_indices += list(
                    rng.choice(label_indices, n_missing, replace=True)
                )
        return Tensor_Corpus(
            *get_all_data(Subset(parent, train_indices)),
            parent=parent,
        )

    if oversample_method.lower() in ('smote', 'adasyn'):
        raise DeprecationWarning(f"Deprecating {oversample_method}")

    raise ValueError(f"Unknown oversample_method '{oversample_method}'.")
