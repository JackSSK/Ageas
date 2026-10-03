#!/usr/bin/env python3
"""Lightning-based neural network classifiers.

Defines :class:`NN_Classifier`, a generic stack of building blocks from
:mod:`ageas.nn.blocks` (MLP/RNN), and :class:`Mixer_Classifier`, a
ResNet-mixer specialisation built on top of it.

Inspired by ``torchvision.models.resnet``
(https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/resnet.py#L124).
"""
import logging
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection
from warnings import warn

from ageas.nn.blocks import Factorized_Residual_Mixer, ResNet_Basic, get_block
from ageas.tool import Basic_Clf_Explainer as Clf_Explainer
from ageas.tool import configure_optimizers as config_optim
from ageas.tool import l1_normalize
from ageas.tool.scores import contrast_classes, score_table

_logger = logging.getLogger(__name__)

__all__ = ['NN_Classifier', 'Mixer_Classifier']

def _with_train_defaults(train_config: dict = None) -> dict:
    """``train_config`` completed with the keys the model itself needs.

    Without a config, :data:`DEFAULT_TRAIN_CONFIG` is used. A given config
    only gets ``loss_reduction`` filled in. Optimizer and scheduler keys it
    leaves out keep the defaults of :func:`~ageas.tool.configure_optimizers`.
    """
    if train_config is None:
        return dict(DEFAULT_TRAIN_CONFIG)
    return {'loss_reduction': 'mean', **train_config}


# Training settings used when no ``train_config`` is given.
DEFAULT_TRAIN_CONFIG = {
    'loss_reduction': 'mean',
    'optimizer': 'adamw',
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
    'momentum': 0.9,
    'scheduler': 'cosine',
    'sch_T_0': 10,
    'sch_T_mult': 2,
    'sch_eta_min': 1e-6,
    'total_steps': 100,
}

NORM_DICT = {
    'BatchNorm1d': nn.BatchNorm1d,
    'GroupNorm': nn.GroupNorm,
    'LayerNorm': nn.LayerNorm,
}


class NN_Classifier(pl.LightningModule):
    """Modular Lightning classifier built from :mod:`ageas.nn.blocks`.

    The model stacks (optional) feature embedding, a configurable cascade
    of blocks defined by ``model_params['block']``, ``block_nums``, and
    ``block_dims``, and a final linear decision layer. With zero blocks the
    network reduces to a single linear layer (logistic regression on the
    embedded features).

    Attributes:
        current_fea_dim: Current feature dimension, updated as blocks are
            stacked.
        embedder: Feature embedding module (linear + norm + ReLU, or
            identity).
        blocks: Stacked block groups as a :class:`~torch.nn.ModuleList`.
        dropout: Dropout layer before the decision layer.
        fc: Final linear decision layer.
        criterion: Cross-entropy loss function.
        accuracy: Multiclass accuracy metric.
        f1: Macro-averaged F1 metric.
        auroc: Macro-averaged AUROC metric.
    """

    def __init__(
        self,
        model_params: dict = None,
        train_config: dict = None,
    ) -> None:
        """Initialize an NN_Classifier.

        Args:
            model_params: Architecture hyper-parameters. Recognized keys:
                ``len_in``, ``inplanes``, ``block_nums``, ``block_dims``,
                ``latent_fea_dim``, ``num_classes``, ``nonlinearity``,
                ``bias``, ``dropout``, ``bidirectional``, ``proj_size``,
                ``norm_layer``, ``block``.
            train_config: Training hyper-parameters. Recognized keys:
                ``loss_reduction``, ``optimizer``, ``learning_rate``,
                ``weight_decay``, ``betas``, ``momentum``, ``scheduler``,
                ``sch_T_0``, ``sch_T_mult``, ``sch_eta_min``,
                ``total_steps``. See :func:`_with_train_defaults` for
                missing keys.
        """
        if model_params is None:
            model_params = {
                'len_in': 20492,
                'inplanes': 64,
                'block_nums': [],
                'block_dims': [],
                'latent_fea_dim': 1000,
                'num_classes': 2,
                'nonlinearity': 'tanh',
                'bias': True,
                'dropout': 0.0,
                'bidirectional': True,
                'proj_size': 0,
                'norm_layer': 'LayerNorm',
            }
        train_config = _with_train_defaults(train_config)

        super().__init__()
        # Use explicit names so save_hyperparameters() reads the current
        # local values (which may have been filled from None defaults above)
        # rather than the None signature defaults, regardless of PL version.
        self.save_hyperparameters('model_params', 'train_config')

        self._build_architecture(model_params)

        self.criterion = nn.CrossEntropyLoss(
            reduction=train_config['loss_reduction']
        )
        self.set_metrics(model_params['num_classes'])

    def sanity_check(self, model_params: dict) -> None:
        """Validate ``model_params`` and resolve block/norm-layer classes.

        Args:
            model_params: Hyper-parameter dict shared across the classifier
                and its blocks. Length-matched fields (``block_nums``,
                ``block_dims``, ``block_planes``,
                ``replace_stride_with_dilation``) are checked for
                consistency, and ``block`` / ``norm_layer`` strings are
                resolved into classes stored on ``self``.

        Raises:
            AssertionError: If length-matched fields are inconsistent.
        """
        if 'block_nums' in model_params:
            num_blocks = len(model_params['block_nums'])
            if 'block_dims' in model_params:
                assert num_blocks == len(model_params['block_dims'])
            if 'block_planes' in model_params:
                assert num_blocks == len(model_params['block_planes'])
            if 'replace_stride_with_dilation' in model_params:
                assert num_blocks == len(model_params['replace_stride_with_dilation'])

        if 'block_layer_type' in model_params and (
            len(set(model_params['block_nums'])) > 1
        ):
            warn(
                'Hidden state (and cell state) will be re-initialized between '
                'blocks due to inconsistent num_layers'
            )

        if 'block' in model_params:
            self.block = get_block(model_params['block'])
        if 'norm_layer' in model_params:
            assert model_params['norm_layer'] in NORM_DICT, \
                f"norm_layer '{model_params['norm_layer']}' not recognised."
            self.norm_layer = NORM_DICT[model_params['norm_layer']]

    def _build_architecture(self, model_params: dict) -> None:
        """Run sanity checks and build the linear embedder, blocks, and head.

        Subclasses override this hook to install a different module stack
        while inheriting the common ``__init__`` scaffolding (hparams,
        criterion, metrics).

        Args:
            model_params: Architecture hyper-parameter dict.
        """
        self.sanity_check(model_params)

        if model_params['latent_fea_dim'] is not None:
            self.current_fea_dim = model_params['latent_fea_dim']
            self.embedder = nn.Sequential(
                nn.Linear(model_params['len_in'], self.current_fea_dim, bias=True),
                self.norm_layer(self.current_fea_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.current_fea_dim = model_params['len_in']
            self.embedder = nn.Identity()

        self.blocks = nn.ModuleList()
        for num_blocks, out_dim in zip(
            model_params['block_nums'], model_params['block_dims']
        ):
            extra = {
                k: v
                for k, v in model_params.items()
                if k not in ('block', 'n_blocks', 'out_dim', 'norm_layer',
                             'block_nums', 'block_dims')
            }
            self.blocks.append(
                self._make_block(
                    block=self.block,
                    n_blocks=num_blocks,
                    out_dim=out_dim,
                    **extra,
                )
            )

        self.dropout = nn.Dropout(p=model_params['dropout'])
        self.fc = nn.Linear(
            model_params['inplanes'] * self.current_fea_dim,
            model_params['num_classes'],
            bias=True,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_metrics(self, num_classes: int) -> None:
        """Attach validation and test metrics: accuracy, macro F1, AUROC.

        Each stage has its own metric objects. They accumulate over every
        batch of an epoch and are computed once over the whole split, so
        small batches (where one batch may hold a single class) don't
        distort F1 or AUROC.

        Args:
            num_classes: Number of label classes.
        """
        def stage_metrics(prefix: str) -> MetricCollection:
            return MetricCollection(
                {
                    'accuracy': Accuracy(num_classes=num_classes, task='multiclass'),
                    'f1': F1Score(
                        num_classes=num_classes, task='multiclass', average='macro'
                    ),
                    'auroc': AUROC(num_classes=num_classes, task='multiclass'),
                },
                prefix=prefix,
            )

        self.vali_metrics = stage_metrics('vali.')
        self.test_metrics = stage_metrics('test.')

    def forward(self, x):
        """Forward pass through embedder, blocks, and decision layer.

        Args:
            x: Input tensor.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.
        """
        x = self.embedder(x)
        for block_group in self.blocks:
            output = block_group(x)
            # RNN-based blocks return (output, h0, c0) tuples
            x = output[0] if isinstance(output, tuple) else output
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    @torch.no_grad()
    def predict(self, x, y=None) -> np.ndarray:
        """Forward ``x`` through the model and return softmax probabilities.

        Args:
            x: Input tensor.
            y: Unused; kept for API compatibility with non-NN classifiers.

        Returns:
            Per-class probability matrix of shape ``(batch, num_classes)``.
        """
        self.eval()
        # The model may sit on another device, e.g. after explaining on GPU.
        out = nn.functional.softmax(self(x.to(self.device)), dim=-1)
        return out.cpu().numpy()

    def configure_optimizers(self):
        """Create the optimizer and the learning rate schedule.

        Returns:
            Output of :func:`ageas.tool.configure_optimizers` configured
            from ``self.hparams.train_config``.
        """
        return config_optim(
            named_parameters=self.named_parameters(),
            **self.hparams.train_config,
        )

    def training_step(self, batch, batch_idx: int = None):
        """Lightning training step. Logs cross-entropy loss as ``CEL``.

        Args:
            batch: Tuple ``(x, y)`` from the dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log_dict({'CEL': loss}, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int = None):
        """Lightning validation step.

        Logs ``vali.CEL``, ``vali.accuracy``, ``vali.f1`` and ``vali.auroc``.

        Args:
            batch: Tuple ``(x, y)`` from the dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Scalar validation loss tensor.
        """
        return self._evaluate(batch, 'vali.', self.vali_metrics)

    def test_step(self, batch, batch_idx: int = None):
        """Lightning test step.

        Logs ``test.CEL``, ``test.accuracy``, ``test.f1`` and
        ``test.auroc``.

        Args:
            batch: Tuple ``(x, y)`` from the dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Scalar test loss tensor.
        """
        return self._evaluate(batch, 'test.', self.test_metrics)

    def _evaluate(self, batch, prefix: str, metrics: MetricCollection):
        """Log the batch-size-weighted loss and update whole-split metrics."""
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log(
            f'{prefix}CEL', loss, on_step=False, on_epoch=True,
            batch_size=len(y), sync_dist=True, prog_bar=True,
        )
        metrics.update(out, y)
        # Logging the collection makes Lightning compute and reset it once
        # per epoch, over every batch of the split.
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def explain(
        self,
        dataset=None,
        exp_sample_limit: int = None,
        verbose: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute per-class feature attributions via Integrated Gradients.

        For every class the dataset is stratified by predicted label, the
        attributions are averaged per feature, L1-normalised, and finally
        contrasted against the maximum of the other classes to highlight
        class-specific contributions.

        Args:
            dataset: Explanation corpus. Must expose ``label_dict``,
                ``features``, and a compatible ``stratify`` interface.
            exp_sample_limit: Optional cap on the number of samples explained
                per class.
            verbose: If ``True``, print per-class progress.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~ageas.tool.Basic_Clf_Explainer`.

        Returns:
            Per-feature, per-class score and standard-deviation table as a
            :class:`~pandas.DataFrame`.
        """
        n_class = (
            len(dataset.label_dict)
            if dataset.label_dict is not None
            else self.hparams.model_params['num_classes']
        )

        # The explainer sets a process-wide matmul precision; restore it after.
        previous_precision = torch.get_float32_matmul_precision()
        try:
            clf_explainer = Clf_Explainer(
                model=self,
                dataset=dataset,
                verbose=verbose,
                **kwargs,
            )

            explain_means: list = []
            explain_stds: list = []

            for i in range(n_class):
                if verbose:
                    _logger.info('Launching Class %d explanation...', i)
                    start = time.time()

                dataloader = clf_explainer.stratify_dataset(
                    class_index=i,
                    sample_limit=exp_sample_limit,
                )

                # Attribute the trained model as it predicts: eval mode keeps
                # dropout off and BatchNorm on its running statistics (which
                # train mode would also overwrite). cuDNN refuses RNN backward
                # passes in eval mode, so it is disabled for the attribution.
                self.eval()
                with torch.backends.cudnn.flags(enabled=False):
                    exp_mean, exp_std = clf_explainer(i, dataloader=dataloader)

                if exp_mean is None or torch.all(torch.isnan(exp_mean)):
                    exp_mean = torch.zeros(len(dataset.features))
                if exp_std is None or torch.all(torch.isnan(exp_std)):
                    exp_std = torch.zeros(len(dataset.features))

                explain_means.append(l1_normalize(exp_mean))
                explain_stds.append(l1_normalize(exp_std))

                if verbose:
                    elapsed = (time.time() - start) / 60
                    _logger.info(
                        'Class %d explanation done: %.2f min', i, elapsed
                    )
        finally:
            torch.set_float32_matmul_precision(previous_precision)

        explain_means = np.array([x.cpu().numpy() for x in explain_means])
        explain_stds = np.array([x.cpu().numpy() for x in explain_stds])

        return score_table(
            dataset.features,
            contrast_classes(explain_means),
            stds=explain_stds,
        )

    def _make_block(
        self,
        block,
        n_blocks: int = 1,
        out_dim: int = 64,
        **kwargs,
    ) -> nn.Sequential:
        """Build a sequential group of blocks.

        Args:
            block: Block class to instantiate.
            n_blocks: Number of blocks to stack.
            out_dim: Output dimension of each block.
            **kwargs: Forwarded to each block's constructor.

        Returns:
            :class:`~torch.nn.Sequential` containing the stacked blocks.
        """
        if n_blocks <= 0:
            return nn.Sequential()

        layers = [
            block(
                in_dim=self.current_fea_dim,
                out_dim=out_dim,
                norm_layer=self.norm_layer,
                **kwargs,
            )
        ]
        self.current_fea_dim = out_dim

        for _ in range(n_blocks - 1):
            layers.append(
                block(
                    in_dim=out_dim,
                    out_dim=out_dim,
                    norm_layer=self.norm_layer,
                    **kwargs,
                )
            )
        return nn.Sequential(*layers)


class Mixer_Classifier(NN_Classifier):
    """ResNet-style mixer classifier.

    Specialisation of :class:`NN_Classifier` that replaces the linear
    embedder with a global 1D convolution and stacks
    :class:`~ageas.nn.blocks.ResNet_Basic` /
    :class:`~ageas.nn.blocks.Factorized_Residual_Mixer` blocks. The minimum
    configuration is one convolutional embedder followed by a linear decision
    layer.

    Attributes:
        current_planes: Current channel count, updated as blocks are built.
        current_dilation: Current dilation factor.
        embedder: Convolutional embedder followed by norm and ReLU.
        maxpool: Max-pooling layer between embedder and residual blocks.
        blocks: Stacked residual block groups.
        avgpool: Adaptive average pooling before the decision layer.
        dropout: Dropout before the decision layer.
        fc: Final linear decision layer.
    """

    def __init__(
        self,
        model_params: dict = None,
        train_config: dict = None,
        **kwargs,
    ) -> None:
        """Initialize a Mixer_Classifier.

        Args:
            model_params: Architecture hyper-parameters. Recognized keys
                (in addition to :class:`NN_Classifier` keys):
                ``latent_planes``, ``global_conv``, ``block_planes``,
                ``zero_init_residual``, ``groups``, ``dilation``,
                ``width_per_group``, ``replace_stride_with_dilation``.
            train_config: Training hyper-parameters (same as
                :class:`NN_Classifier`).
            **kwargs: Ignored extra arguments.
        """
        if model_params is None:
            model_params = {
                'len_in': 20492,
                'inplanes': 64,
                'latent_planes': 64,
                'latent_fea_dim': 1000,
                'global_conv': True,
                'num_classes': 1000,
                'block': 'ResNet_Basic',
                'block_nums': [2, 2, 2, 2],
                'block_planes': [64, 128, 256, 512],
                'dropout': 0.0,
                'zero_init_residual': False,
                'groups': 1,
                'dilation': 1,
                'width_per_group': 64,
                'replace_stride_with_dilation': False,
                'norm_layer': 'BatchNorm1d',
            }
        train_config = _with_train_defaults(train_config)

        super().__init__(model_params=model_params, train_config=train_config)

    def _build_architecture(self, model_params: dict) -> None:
        """Build the convolutional embedder, residual blocks, and head.

        Overrides :meth:`NN_Classifier._build_architecture` to install the
        ResNet-mixer module stack while inheriting the parent's hparam,
        criterion, and metric setup.

        Args:
            model_params: Architecture hyper-parameter dict.
        """
        if isinstance(model_params['replace_stride_with_dilation'], bool):
            model_params['replace_stride_with_dilation'] = [
                model_params['replace_stride_with_dilation']
            ] * len(model_params['block_nums'])

        self.sanity_check(model_params)

        self.current_planes = model_params['latent_planes']
        self.current_dilation = model_params['dilation']

        self.embedder = nn.Sequential(
            nn.Conv1d(
                model_params['inplanes'],
                model_params['latent_planes'] * model_params['latent_fea_dim'] * 2,
                kernel_size=model_params['len_in'],
                stride=1,
                padding=0,
                bias=False,
            ),
            self.norm_layer(
                model_params['latent_planes'] * model_params['latent_fea_dim'] * 2
            ),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(len(model_params['block_nums'])):
            self.blocks.append(
                self._make_block(
                    block=self.block,
                    norm_layer=self.norm_layer,
                    planes=model_params['block_planes'][i],
                    n_blocks=model_params['block_nums'][i],
                    stride=1,
                    dilate=model_params['replace_stride_with_dilation'][i],
                )
            )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(model_params['dropout'])
        self.fc = nn.Linear(self.current_planes, model_params['num_classes'])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch so that the
        # residual branch starts with zeros and behaves like an identity.
        # This improves accuracy by 0.2–0.3% per https://arxiv.org/abs/1706.02677
        if model_params.get('zero_init_residual'):
            for m in self.modules():
                if isinstance(m, Factorized_Residual_Mixer):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNet_Basic):
                    nn.init.constant_(m.bn2.weight, 0)

        self.explainer = None

    def _make_block(
        self,
        block,
        norm_layer=nn.BatchNorm1d,
        planes: int = 64,
        n_blocks: int = 1,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Build a sequential group of residual blocks with optional downsampling.

        Args:
            block: Block class to instantiate.
            norm_layer: Normalisation layer class.
            planes: Base number of output channels.
            n_blocks: Number of blocks to stack.
            stride: Stride for the first block's spatial convolution.
            dilate: If ``True``, replace stride with dilation.

        Returns:
            :class:`~torch.nn.Sequential` containing the stacked blocks.
        """
        if n_blocks <= 0:
            return nn.Sequential()

        downsample = None
        previous_dilation = self.current_dilation
        if dilate:
            self.current_dilation *= stride
            stride = 1

        if stride != 1 or self.current_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.current_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * block.expansion),
            )

        mp = self.hparams.model_params
        layers = [
            block(
                len_in=mp['latent_fea_dim'],
                inplanes=self.current_planes,
                planes=planes,
                global_conv=mp['global_conv'],
                stride=stride,
                downsample=downsample,
                groups=mp['groups'],
                base_width=mp['width_per_group'],
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        ]
        self.current_planes = planes * block.expansion

        for _ in range(1, n_blocks):
            layers.append(
                block(
                    len_in=mp['latent_fea_dim'],
                    inplanes=self.current_planes,
                    planes=planes,
                    global_conv=mp['global_conv'],
                    groups=mp['groups'],
                    base_width=mp['width_per_group'],
                    dilation=self.current_dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through embedder, max-pool, blocks, and decision layer.

        Args:
            x: Input tensor.

        Returns:
            Logit tensor of shape ``(batch, num_classes)``.
        """
        x = self.embedder(x)
        x = x.reshape(
            x.shape[0],
            -1,
            self.hparams.model_params['latent_fea_dim'] * 2,
        )
        x = self.maxpool(x)

        for block_group in self.blocks:
            x = block_group(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
