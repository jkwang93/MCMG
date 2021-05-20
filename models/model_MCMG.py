# _*_ encoding: utf-8 _*_


__author__ = 'wjk'
__date__ = '2020/6/30 10:32'

import math
import numpy as np
import torch
import torch.nn as nn
# from einops import rearrange
import torch.nn.functional as F

from .model_Transformer import GPTDecoderLayer, GPTDecoder
from MCMG_utils.utils import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class decoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        # self.embed_src = nn.Embedding(vocab_size, d_model)
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionEncoding(d_model, pos_dropout, max_seq_length)

        decoder_layers = GPTDecoderLayer(d_model, nhead, dim_feedforward, trans_dropout)
        self.transformer_encoder = GPTDecoder(decoder_layers, num_decoder_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, con_token, tgt_key_padding_mask, tgt_mask):
        # (S ,N ,EMBEDDING_DIM) 所以先要转置

        con_token = con_token.transpose(0, 1)

        # src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        # src = rearrange(src, 'n s -> s n')
        # tgt = rearrange(tgt, 'n t -> t n')

        # src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        # add conditional token embedding

        # a = self.embed_tgt(con_token).sum(dim=0).unsqueeze(0)
        # b = self.embed_tgt(tgt)
        # c = a + b

        tgt = self.pos_enc(
            (self.embed_tgt(tgt) + self.embed_tgt(con_token).sum(dim=0).unsqueeze(0)) * math.sqrt(self.d_model))

        output = self.transformer_encoder(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)

        return self.fc(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=140):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 可学习位置编码
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=140):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        a = weight[:x.size(0), :]
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class transformer_RL():
    def __init__(self, voc, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout):
        self.decodertf = decoderTransformer(voc.vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward,
                                            max_seq_length, pos_dropout, trans_dropout)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.decodertf.to(self.device)
        self.voc = voc
        self._nll_loss = nn.NLLLoss(ignore_index=0, reduction="none")

    def likelihood(self, target):
        """
               Retrieves the likelihood of a given sequence

               Args:
                   target: (batch_size * sequence_lenght) A batch of sequences

               Outputs:
                   log_probs : (batch_size) Log likelihood for each example*
                   entropy: (batch_size) The entropies for the sequences. Not
                                         currently used.
        """
        batch_size, seq_length = target.size()

        # print(self.voc.vocab)
        # print(self.voc.vocab['EOS'])

        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        # split train and target and con_token
        con_token = target[:, 0:4]
        x = torch.cat((start_token, target[:, 4:-1]), 1)
        y_target = target[:, 4:].contiguous().view(-1)

        # 去掉token长度
        seq_length = seq_length - len(con_token[0])

        # log_probs = Variable(torch.zeros(batch_size))
        # entropy = Variable(torch.zeros(batch_size))

        logits = Prior_train_forward(self.decodertf, x, con_token)

        # print(logits.shape)

        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        log_probs = criterion(logits.view(-1, self.voc.vocab_size), y_target)
        # loss值
        mean_log_probs = log_probs.mean()

        # print('gooo',log_probs.shape)
        log_probs = log_probs.view(-1, seq_length)
        # 求和，得到每个分子的loss值
        log_probs_each_molecule = log_probs.sum(dim=1)

        return mean_log_probs, log_probs_each_molecule

    def sample(self, batch_size, max_length=140, con_token_list= ['is_JNK3', 'is_GSK3', 'high_QED', 'good_SA']):
        """
               Sample a batch of sequences

               Args:
                   batch_size : Number of sequences to sample
                   max_length:  Maximum length of the sequences

               Outputs:
               seqs: (batch_size, seq_length) The sampled sequences.
               log_probs : (batch_size) Log likelihood for each sequence.
               entropy: (batch_size) The entropies for the sequences. Not
                                       currently used.
       """

        # conditional token
        con_token_list = Variable(self.voc.encode(con_token_list))

        con_tokens = Variable(torch.zeros(batch_size, len(con_token_list)).long())

        for ind, token in enumerate(con_token_list):
            con_tokens[:, ind] = token

        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        input_vector = start_token
        # print(batch_size)

        sequences = start_token
        log_probs = Variable(torch.zeros(batch_size))
        # log_probs1 = Variable(torch.zeros(batch_size))

        finished = torch.zeros(batch_size).byte()

        finished = finished.to(self.device)

        for step in range(max_length):
            logits = sample_forward_model(self.decodertf, input_vector, con_tokens)

            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            log_prob = F.log_softmax(logits_step, dim=1)

            input_vector = torch.multinomial(prob, 1)

            # need to concat prior words as the sequences and input 记录下每一步采样
            sequences = torch.cat((sequences, input_vector), 1)


            log_probs += self._nll_loss(log_prob, input_vector.view(-1))
            # log_probs1 += NLLLoss(log_prob, input_vector.view(-1))
            # print(log_probs1==-log_probs)




            EOS_sampled = (input_vector.view(-1) == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)

            if torch.prod(finished) == 1:
                # print('End')
                break

            # because there are no hidden layer in transformer, so we need to append generated word in every step as the input_vector
            input_vector = sequences

        return sequences[:, 1:].data, log_probs

    def generate(self, batch_size, max_length=140, con_token_list= ['is_JNK3', 'is_GSK3', 'high_QED', 'good_SA']):
        """
               Sample a batch of sequences

               Args:
                   batch_size : Number of sequences to sample
                   max_length:  Maximum length of the sequences

               Outputs:
               seqs: (batch_size, seq_length) The sampled sequences.
               log_probs : (batch_size) Log likelihood for each sequence.
               entropy: (batch_size) The entropies for the sequences. Not
                                       currently used.
       """
        # conditional token
        con_token_list = Variable(self.voc.encode(con_token_list))

        con_tokens = Variable(torch.zeros(batch_size, len(con_token_list)).long())

        for ind, token in enumerate(con_token_list):
            con_tokens[:,ind] = token


        start_token = Variable(torch.zeros(batch_size, 1).long())

        start_token[:] = self.voc.vocab['GO']

        input_vector = start_token
        # print(batch_size)

        sequences = start_token
        # log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        # entropy = Variable(torch.zeros(batch_size))

        finished = finished.to(self.device)

        for step in range(max_length):
            # print(step)
            logits = sample_forward_model(self.decodertf, input_vector, con_tokens)

            logits_step = logits[:, step, :]

            prob = F.softmax(logits_step, dim=1)
            # log_prob = F.log_softmax(logits_step, dim=1)

            input_vector = torch.multinomial(prob, 1)

            # need to concat prior words as the sequences and input 记录下每一步采样
            sequences = torch.cat((sequences, input_vector), 1)
            EOS_sampled = (input_vector.view(-1) == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            '''一次性计算所有step的nll'''
            if torch.prod(finished) == 1:
                # print('End')
                break

            # because there are no hidden layer in transformer, so we need to append generated word in every step as the input_vector
            input_vector = sequences

        return sequences[:, 1:].data


def Prior_train_forward(model, tgt, con_token):
    tgt = torch.tensor(tgt)

    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    output = model(tgt, con_token, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask)

    del tgt, tgt_mask

    return output.squeeze(1)


def sample_forward_model(model, tgt, con_token):
    # src = torch.tensor(src).unsqueeze(0).long().to('cuda')
    # tgt = torch.tensor(tgt)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    output = model(tgt, con_token, tgt_key_padding_mask=None,
                   tgt_mask=tgt_mask)
    # print(output.shape)

    return output


# 从一开始就不用遮住conditional token，mask从第len(conditonal)+len(start)之后开始
def gen_nopeek_mask(length):
    # mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask.to(device)

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).to(device)
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
