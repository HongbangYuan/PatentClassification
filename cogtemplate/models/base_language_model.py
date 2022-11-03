from cogtemplate.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class FullyConnectedForLM(BaseModel):
    def __init__(self,n_token,embedding_dim,length,hidden_size):
        super(FullyConnectedForLM, self).__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        self.encode = nn.Linear(embedding_dim * length,hidden_size)
        self.decode = nn.Linear(hidden_size,length * n_token)
        self.length = length
        self.vocab_size = n_token


    def forward(self, batch):
        word_ids,word_length = batch["word_ids"],batch["word_length"]
        batch_size = word_length.shape[0]
        word_ids_embedding = self.embedding(word_ids)
        hidden = self.encode(word_ids_embedding.reshape(batch_size,-1))
        logits = self.decode(hidden)
        return logits.reshape(batch_size,self.length,-1)

    def loss(self, batch, loss_function):
        logits = self.forward(batch)
        label_ids = batch["label_ids"]
        loss = loss_function(logits.reshape(-1, self.vocab_size), label_ids.reshape(-1))
        return loss

    def evaluate(self, batch, metric_function):
        label_ids = batch["label_ids"]
        logits = self.forward(batch)
        metric_function.evaluate(logits.reshape(-1, self.vocab_size), label_ids.reshape(-1))

    def predict(self, batch):
        pass

class GRUForLM(BaseModel):
    def __init__(self,n_token,embedding_dim,hidden_size):
        super(GRUForLM, self).__init__()
        self.embedding = nn.Embedding(n_token,embedding_dim)
        self.gru = MyGRU(embedding_dim,hidden_size,)
        self.decoder = nn.Linear(hidden_size,n_token)
        self.vocab_size = n_token

    def forward(self, batch):
        word_ids,word_length = batch["word_ids"],batch["word_length"]
        word_ids_embedding = self.embedding(word_ids).transpose(0,1)
        h,h_n = self.gru(word_ids_embedding,word_length.cpu().numpy(),need_h=True)
        logits = self.decoder(h.transpose(0,1))
        return logits

    def loss(self, batch, loss_function):
        label_ids = batch["label_ids"][:,:torch.max(batch["word_length"]).item()]
        logits = self.forward(batch)
        loss = loss_function(logits.reshape(-1,self.vocab_size),label_ids.reshape(-1))
        return loss

    def predict(self, batch):
        pass

    def evaluate(self, batch, metric_function):
        label_ids = batch["label_ids"][:,:torch.max(batch["word_length"]).item()]
        logits = self.forward(batch)
        metric_function.evaluate(logits.reshape(-1,self.vocab_size),label_ids.reshape(-1))


class TransformerForLM(BaseModel):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask) :
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def loss(self, batch, loss_function):
        src,src_mask = batch["word_ids"],batch["src_mask"]
        label = batch["label_ids"]
        output = self.forward(src.T,src_mask[0])
        output = output.transpose(0,1)
        n_vocab = output.size(-1)
        loss = loss_function(output.reshape(-1,n_vocab), label.reshape(-1))
        return loss

    def evaluate(self, batch, metric_function):
        src,src_mask = batch["word_ids"],batch["src_mask"]
        label = batch["label_ids"]
        output = self.forward(src.T,src_mask[0])
        output = output.transpose(0,1)
        n_vocab = output.size(-1)
        metric_function.evaluate(output.reshape(-1,n_vocab), label.reshape(-1))

    def predict(self, batch):
        src, src_mask = batch["word_ids"], batch["src_mask"]
        pred = self.forward(src,src_mask)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Coding: utf-8
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


import torch
import os


_VF = torch._C._VariableFunctions
F_GRUCell = _VF.gru_cell


def sortSequence(data, length):
    shape = data.shape
    len, fsize = shape[0], shape[-1]
    data = data.reshape(len, -1, fsize)
    batch_size = data.shape[1]
    length = length.reshape(-1)

    zero_num = np.sum(length == 0)
    memo = list(reversed(np.argsort(length).tolist()))[:batch_size - zero_num]
    res = torch.zeros([data.shape[0], batch_size - zero_num, data.shape[-1]],dtype=data.dtype,device=data.device)
    for i, idx in enumerate(memo):
        res[:, i, :] = data[:, idx, :]
    return res, sorted(length, reverse=True)[: batch_size - zero_num], (shape, memo, zero_num)


def sortSequenceByMemo(data, memo):
    data = data.reshape(-1, data.shape[-1])
    batch_size = data.shape[0]
    shape, memo, zero_num = memo
    res = torch.zeros([batch_size - zero_num, data.shape[-1]], dtype=data.dtype, device=data.device)
    for i, idx in enumerate(memo):
        res[i, :] = data[idx, :]
    return res


def revertSequence(data, memo, isseq=False):
    shape, memo, zero_num = memo
    if isseq:
        res = torch.zeros([data.shape[0], data.shape[1] + zero_num, data.shape[2]],dtype=data.dtype,device=data.device)
        for i, idx in enumerate(memo):
            res[:, idx, :] = data[:, i, :]
        return res.reshape(*((res.shape[0],) + shape[1:-1] + (res.shape[-1],)))
    else:
        res = torch.zeros([data.shape[0] + zero_num, data.shape[1]], dtype=data.dtype,device=data.device)
        for i, idx in enumerate(memo):
            res[idx, :] = data[i, :]
        return res.reshape(*(shape[1:-1] + (res.shape[-1],)))


def flattenSequence(data, length):
    arr = []
    for i in range(length.size):
        arr.append(data[0:length[i], i])
    return torch.cat(arr, dim=0)


def copySequence(data, length):  # for BOW loss
    arr = []
    for i in range(length.size):
        arr.append(data[i].repeat(length[i], 1))
    return torch.cat(arr, dim=0)


def maskedSoftmax(data, length):
    mask = torch.tensor((np.expand_dims(np.arange(data.shape[0]), 1) < np.expand_dims(length, 0)).astype(int),dtype=data.dtype,device=data.device)
    return data.masked_fill(mask == 0, -1e9).softmax(dim=0)


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, bidirectional=False, initpara=True, attn_decode=False,
                 post_size=None):
        super(MyGRU, self).__init__()

        self.input_size, self.hidden_size, self.layers, self.bidirectional = \
            input_size, hidden_size, layers, bidirectional
        self.GRU = GRU(input_size, hidden_size, layers, bidirectional=bidirectional)
        self.initpara = initpara
        if initpara:
            if bidirectional:
                self.h_init = Parameter(torch.Tensor(2 * layers, 1, hidden_size))
            else:
                self.h_init = Parameter(torch.Tensor(layers, 1, hidden_size))
        self.reset_parameters()

        if attn_decode:
            self.attn_query = nn.Linear(hidden_size, post_size)

    def reset_parameters(self):
        if self.initpara:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            self.h_init.data.uniform_(-stdv, stdv)

    def getInitialParameter(self, batch_size):
        return self.h_init.repeat(1, batch_size, 1)

    def forward(self, incoming, length, h_init=None, need_h=False, attn_decode=False, post=None, post_length=None):
        if not attn_decode:
            sen_sorted, length_sorted, memo = sortSequence(incoming, length)
            left_batch_size = sen_sorted.shape[-2]
            sen_packed = pack_padded_sequence(sen_sorted, length_sorted)
            if h_init is None:
                h_init = self.getInitialParameter(left_batch_size)
            else:
                h_shape = h_init.size()
                h_init = sortSequenceByMemo(h_init, memo)
                h_init = h_init.reshape(h_shape)
                if h_init.dim() < 3:
                    h_init = torch.unsqueeze(h_init, 0)

            h, h_n = self.GRU(sen_packed, h_init)
            h_n = h_n.transpose(0, 1).reshape(left_batch_size, -1)
            h_n = revertSequence(h_n, memo)
            if need_h:
                h = pad_packed_sequence(h)[0]
                h = revertSequence(h, memo, True)
                return h, h_n
            else:
                return h_n

        else:
            batch_size = incoming.shape[1]
            seqlen = incoming.shape[0]
            if h_init is None:
                h_init = self.getInitialParameter(batch_size)
            else:
                h_init = torch.unsqueeze(h_init, 0)
            h_now = h_init[0]
            hs = []
            attn_weights = []

            for i in range(seqlen):
                query = self.attn_query(h_now)
                attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
                context = (attn_weight.unsqueeze(-1) * post).sum(0)
                h_now = self.cell_forward(torch.cat([incoming[i], context], dim=-1), h_now) * Tensor(
                    (length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

                hs.append(h_now)
                attn_weights.append(attn_weight)

            return torch.stack(hs), h_now

    def cell_forward(self, incoming, h):
        return F_GRUCell(
            incoming, h,
            self.GRU.weight_ih_l0, self.GRU.weight_hh_l0,
            self.GRU.bias_ih_l0, self.GRU.bias_hh_l0
        )