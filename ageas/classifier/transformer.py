#!/usr/bin/env python3
"""
A playground file about using Transformer type clf
However, it did not show better performace yet
Probably not a good idea

author: jy, nkmtmsys
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer
import ageas.classifier as classifier


class Positional_Encoding(nn.Module):
    """
    Inject some information about the relative or absolute position
    of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings,
    so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the position and i is the embed idx)

    :param d_model: the embed dim (required).
    :param dropout: the dropout value (default=0.1).

    Examples:
        >>> pos_encoder = Positional_Encoding(d_model)
    """

    def __init__(self, d_model, dropout = 0.1):
        super(Positional_Encoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(d_model, d_model)
        position = torch.arange(0, d_model, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                    (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class Transformer(nn.Module):
    """
    Container module with an encoder, a transformer module, and a decoder.
    """
    def __init__(self,
                 id = 'transformer', # model id
                 num_features = 1024, # the number of expected features
                 has_mask = True, # whether using mask or not
                 emsize = 512, # size after encoder
                 nhead = 8, # number of heads in the multiheadattention models
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 dim_feedforward = 512, # number of hidden units per layer
                 dropout = 0.5, # dropout ratio
                 activation = 'relu',
                 layer_norm_eps = 1e-05,
                 learning_rate = 0.1,
                 n_class = 2, # number of class for classification
                ):
        super(Transformer, self).__init__()
        self.id = id
        self.model_type = 'Transformer'
        self.emsize = emsize
        self.has_mask = has_mask
        self.embed = nn.Linear(num_features, emsize)
        # self.pos_encoder = Positional_Encoding(emsize, dropout)

        encoder_layer = TransformerEncoderLayer(
            emsize, nhead, dim_feedforward, dropout, activation, layer_norm_eps
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            emsize, nhead, dim_feedforward, dropout, activation, layer_norm_eps
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self.decision = nn.Linear(emsize, n_class)
        # original optimizer is Adam
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss()
        # init_weights
        initrange = 0.1
        nn.init.zeros_(self.decision.bias)
        nn.init.uniform_(self.decision.weight, -initrange, initrange)

    def _triangular_subsequent_mask(self, sz, diagonal = -1):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=diagonal)

    def _square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        return mask.float(
            ).masked_fill(mask==0,float('-inf')).masked_fill(mask==1,float(0.0))

    def _reverse_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        return mask.float(
            ).masked_fill(mask==0,float(0.0)).masked_fill(mask==1,float('-inf'))

    def forward(self, input):
        if self.has_mask:
            mask = self._square_subsequent_mask(len(input))
        else:
            mask = None
        input = self.embed(input)
        # input = self.pos_encoder(input)
        output = self.encoder(input, mask)
        output = self.decoder(input, output)
        output = torch.flatten(output, start_dim = 1)
        output = func.softmax(self.decision(output), dim = -1)
        return output


class Make(classifier.Make_Template):
    """
    Analysis the performances of Transformer based approaches
    with different hyperparameters
    Find the top settings to build
    """
    # Perform classifier training process for given times
    def train(self, dataSets, test_split_set):
        num_features = len(dataSets.dataTest[0])
        for id in self.configs:
            model = Transformer(id, num_features, **self.configs[id]['config'])
            epoch = self.configs[id]['epoch']
            batch_size = self.configs[id]['batch_size']
            self._train_torch(epoch, batch_size, model, dataSets)
            # local test
            model_record = self._evaluate_torch(
                model,
                dataSets.dataTest,
                dataSets.labelTest,
                test_split_set
            )
            self.models.append(model_record)


""" For test """
# if __name__ == '__main__':
#     param = {
#                 'has_mask': True,
#                 'emsize': 512,
#                 'nhead': 8,
#                 'nhid': 200,
#                 'nlayers': 2,
#                 'dropout': 0.5,
#                 'learning_rate': 0.1
#             }
#     data = torch.rand((3,1,22090))
#     model = Transformer(id = 'a', num_features = 22090, **param)
#     model.train()
#     if model.optimizer is not None: model.optimizer.zero_grad()
#     out = model(data)
#     print(out)
#     loss = model.loss_func(out, torch.randint(0,1,(3,)))
#     loss.backward()
#     if model.optimizer is not None: model.optimizer.step()
