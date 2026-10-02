#!/usr/bin/env python3
"""Integrated-gradients based classifier explainer.

Defines :class:`Basic_Clf_Explainer`, the default attribution backend used
by :class:`~ageas.nn.NN_Classifier` to produce per-class feature importance
scores via :class:`captum.attr.IntegratedGradients`.
"""
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Subset

from ageas.tool.corpus_loader import decomp_batch

_logger = logging.getLogger(__name__)


class Basic_Clf_Explainer:
    """Default classifier explainer backed by Captum's Integrated Gradients.

    Wraps a classification model and a dataset together so that
    :meth:`stratify_dataset` can keep only the samples the model predicts
    correctly for each target class, and :meth:`__call__` then computes
    per-feature attribution means and standard deviations on those samples.

    Attributes:
        dataset: Dataset providing ``(x, y)`` samples.
        n_dataloader_workers: Worker processes for the explanation dataloader.
        verbose: If ``True``, print per-batch progress.
        exp_method: Numerical integration method for Integrated Gradients.
        exp_step: Number of integration steps.
        exp_batch_size: Batch size for the explanation dataloader.
        device: Device string (``'cpu'`` or ``'cuda'``).
        model: Model moved to ``device``.
        explainer: Captum :class:`~captum.attr.IntegratedGradients` instance.
    """

    def __init__(
        self,
        model,
        dataset=None,
        multiply_by_inputs: bool = True,
        n_dataloader_workers: int = 10,
        device: str = 'cuda',
        precision: str = 'medium',
        verbose: bool = False,
        exp_method: str = 'gausslegendre',
        exp_step: int = 100,
        exp_batch_size: int = 1,
    ) -> None:
        """Initialize a Basic_Clf_Explainer.

        Args:
            model: Trained classifier to explain. Will be moved to ``device``.
            dataset: Dataset providing ``(x, y)`` samples. Must implement
                ``stratify(class_label)``.
            multiply_by_inputs: Forwarded to
                :class:`captum.attr.IntegratedGradients`.
            n_dataloader_workers: Worker processes for the explanation
                dataloader.
            device: Device to run the model on. ``None`` auto-selects CUDA
                when available.
            precision: Forwarded to
                :func:`torch.set_float32_matmul_precision`.
            verbose: If ``True``, print per-batch progress.
            exp_method: Numerical integration method for Integrated
                Gradients.
            exp_step: Number of integration steps.
            exp_batch_size: Batch size for the explanation dataloader.
        """
        torch.set_float32_matmul_precision(precision)
        self.dataset = dataset
        self.n_dataloader_workers = n_dataloader_workers
        self.verbose = verbose
        self.exp_method = exp_method
        self.exp_step = exp_step
        self.exp_batch_size = exp_batch_size

        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.model = model.to(self.device)
        self.explainer = IntegratedGradients(
            self.model,
            multiply_by_inputs=multiply_by_inputs,
        )

    def __call__(
        self,
        class_index: int = None,
        dataloader: DataLoader = None,
    ) -> tuple:
        """Run Integrated Gradients on every batch and aggregate the result.

        Args:
            class_index: Target class for the attribution.
            dataloader: Explanation dataloader, typically produced by
                :meth:`stratify_dataset`.

        Returns:
            Tuple ``(mean, std)`` of the attribution scores along the sample
            dimension. Both elements are ``None`` if the dataloader is empty.
        """
        answer = None
        for idx, batch in enumerate(dataloader):
            start_time = time.time()
            if self.verbose:
                _logger.debug('\tProcessing batch %d...', idx + 1)

            data, _ = decomp_batch(batch)
            scores = self.explainer.attribute(
                data.to(self.device),
                baselines=torch.zeros_like(data).to(self.device),
                target=class_index,
                additional_forward_args=None,
                n_steps=self.exp_step,
                method=self.exp_method,
                internal_batch_size=None,
                return_convergence_delta=False,
            )

            answer = scores if answer is None else torch.cat((answer, scores), dim=0)

            if self.verbose:
                elapsed = (time.time() - start_time) / 60
                _logger.debug('\t\tFinished batch %d: %.2f min', idx + 1, elapsed)
            torch.cuda.empty_cache()

        if answer is None:
            return None, None

        answer = torch.reshape(answer, (answer.shape[0], -1))
        assert len(answer.shape) == 2
        return torch.mean(answer, dim=0), torch.std(answer, dim=0)

    @torch.no_grad()
    def stratify_dataset(
        self,
        class_index: int = None,
        sample_limit: int = None,
    ) -> DataLoader:
        """Build an explanation dataloader for ``class_index``.

        Per sample the explainer records the cross-entropy loss (when labels
        are available), the probability gap to the second-best class, and
        the predicted class. Only samples that the model already classifies
        correctly are kept; if ``sample_limit`` is provided the kept samples
        are uniformly sub-sampled in ascending-loss order.

        Args:
            class_index: Target class to stratify on.
            sample_limit: Optional cap on the number of explanation samples.

        Returns:
            :class:`~torch.utils.data.DataLoader` over the curated
            explanation subset.
        """
        t_model = self.model.to(self.device)
        t_model.eval()

        strat_dataset = self.dataset.stratify(class_index)
        dataloader = DataLoader(
            strat_dataset,
            num_workers=self.n_dataloader_workers,
            batch_size=1,
            shuffle=False,
        )

        record: dict = {'loss': [], 'gap': [], 'decision': []}

        for batch in dataloader:
            x, y = decomp_batch(batch)
            x = x.to(t_model.device)

            out = t_model(x).detach().cpu()
            prob_out = nn.functional.softmax(out, dim=-1)

            # Measure the gap between the predicted class and the query class
            gap = prob_out[:, class_index] - torch.max(
                torch.cat(
                    [prob_out[:, :class_index], prob_out[:, class_index + 1:]],
                    dim=-1,
                ),
                dim=-1,
            ).values.item()
            record['gap'].append(gap)
            record['decision'].append(torch.argmax(prob_out, dim=-1).item())

            if y is not None:
                record['loss'].append(t_model.criterion(out, y).item())

        torch.cuda.empty_cache()
        if len(record['loss']) == 0:
            del record['loss']

        # The index of ``rec`` is each sample's position in ``strat_dataset``
        # (not in the full dataset), so subsets must be taken from it.
        rec = pd.DataFrame(record)
        rec = rec[rec['decision'] == class_index]
        explain_data = Subset(strat_dataset, rec.index.tolist())

        if sample_limit is not None and len(rec) > 0:
            sample_limit = min(sample_limit, len(rec))
            sort_key = 'loss' if 'loss' in rec.columns else 'gap'
            indices = (
                rec.sort_values(sort_key)
                .iloc[
                    [
                        int(i * (len(rec) - 1) / max(sample_limit - 1, 1))
                        for i in range(sample_limit)
                    ]
                ]
                .index.tolist()
            )
            explain_data = Subset(strat_dataset, indices)

        return DataLoader(
            explain_data,
            num_workers=self.n_dataloader_workers,
            batch_size=self.exp_batch_size,
            shuffle=False,
        )
