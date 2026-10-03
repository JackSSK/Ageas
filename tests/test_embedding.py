"""Per-modality embedding stage and token RNN units."""
import unittest

import torch
import torch.nn as nn

from ageas.nn import NN_Classifier
from ageas.nn.blocks import RNN_Residual_Encoder
from ageas.nn.embedding import Modality_Embedding

MLP = {'block': 'Residual_Encoder', 'block_nums': [1], 'block_dims': [8],
       'latent_fea_dim': 16, 'dropout': 0.0, 'norm_layer': 'LayerNorm'}
RNN = {'block': 'RNN_Residual_Encoder', 'block_layer_type': 'LSTM',
       'block_nums': [1], 'block_dims': [8], 'latent_fea_dim': 16,
       'dropout': 0.0, 'bidirectional': True, 'norm_layer': 'LayerNorm'}


def build(model_params, n_genes=20, n_modalities=1, n_classes=3):
    return NN_Classifier(model_params={
        **model_params, 'num_classes': n_classes,
        'len_in': n_genes, 'inplanes': n_modalities,
    })


class EmbeddingTest(unittest.TestCase):

    def test_single_modality_single_token_is_todays_embedder(self):
        torch.manual_seed(0)
        reference = nn.Sequential(nn.Linear(20, 16), nn.LayerNorm(16), nn.ReLU())
        embedding = Modality_Embedding(
            n_modalities=1, len_in=20, seq_len=1, token_dim=16,
            embedder='mlp', norm_layer=nn.LayerNorm,
        )
        own = list(embedding.parameters())
        ref = list(reference.parameters())
        self.assertEqual([p.shape for p in own], [p.shape for p in ref])
        with torch.no_grad():
            for p, q in zip(own, ref):
                p.copy_(q)
        x = torch.randn(5, 1, 20)
        torch.testing.assert_close(embedding(x), reference(x))

    def test_modalities_are_summed_per_token(self):
        embedding = Modality_Embedding(
            n_modalities=2, len_in=12, seq_len=3, token_dim=4, embedder='linear',
        )
        with torch.no_grad():
            for p, q in zip(embedding.embedders[1].parameters(),
                            embedding.embedders[0].parameters()):
                p.copy_(q)
        x = torch.randn(5, 1, 12)
        single = Modality_Embedding(
            n_modalities=1, len_in=12, seq_len=3, token_dim=4, embedder='linear',
        )
        with torch.no_grad():
            for p, q in zip(single.parameters(), embedding.embedders[0].parameters()):
                p.copy_(q)
        torch.testing.assert_close(embedding(x.repeat(1, 2, 1)), 2 * single(x))

    def test_without_embedder_tokens_are_raw_padded_chunks(self):
        embedding = Modality_Embedding(n_modalities=2, len_in=10, seq_len=4, token_dim=None)
        x = torch.arange(20.0).reshape(1, 2, 10)
        tokens = embedding(x)
        self.assertEqual(tuple(tokens.shape), (1, 4, 3))
        torch.testing.assert_close(tokens[0, 0], x[0, 0, :3] + x[0, 1, :3])
        torch.testing.assert_close(tokens[0, 3], torch.tensor([9.0 + 19.0, 0.0, 0.0]))


class TokenRNNTest(unittest.TestCase):

    def test_rnn_units_read_eight_tokens_by_default(self):
        model = build(RNN)
        seen = []
        lstm = next(m for m in model.modules() if isinstance(m, nn.LSTM))
        lstm.register_forward_hook(lambda m, inputs, out: seen.append(inputs[0].shape))
        model(torch.randn(4, 1, 20))
        self.assertEqual(tuple(seen[0]), (4, 8, 2))

    def test_rnn_output_depends_on_token_order(self):
        torch.manual_seed(0)
        block = RNN_Residual_Encoder(in_dim=4, out_dim=6, block_layer_type='GRU')
        tokens = torch.randn(2, 5, 4)
        forward, _, _ = block(tokens)
        backward, _, _ = block(tokens.flip(1))
        self.assertFalse(torch.allclose(forward, backward.flip(1)))

    def test_block_num_layer_sets_rnn_depth(self):
        model = build({**RNN, 'block_num_layer': 2})
        lstm = next(m for m in model.modules() if isinstance(m, nn.LSTM))
        self.assertEqual(lstm.num_layers, 2)

    def test_multimodal_mlp_runs_on_one_fused_vector(self):
        model = build(MLP, n_modalities=2)
        self.assertEqual(tuple(model(torch.randn(4, 2, 20)).shape), (4, 3))
        self.assertEqual(model.fc.in_features, 8)

    def test_invalid_sequence_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'seq_len'):
            build({**RNN, 'seq_len': 5})  # 16 is not divisible by 5
        with self.assertRaisesRegex(ValueError, 'seq_len'):
            build({**MLP, 'seq_len': 4})  # MLP units read a single token


if __name__ == '__main__':
    unittest.main()
