#!/usr/bin/env python
import argparse

import torch
import time

from models.model_MCMG import transformer_RL
from MCMG_utils.data_structs import Vocabulary
from MCMG_utils.utils import seq_to_smiles
import pandas as pd




def Transformer_generator(restore_prior_from='output/Prior.ckpt',
                          save_file='test.csv',
                          batch_size=128,
                          n_steps=5000,


                          ):
    voc = Vocabulary(init_from_file="data/Voc_RE1")

    start_time = time.time()

    Prior = transformer_RL(voc, d_model, nhead, num_decoder_layers,
                           dim_feedforward, max_seq_length,
                           pos_dropout, trans_dropout)

    Prior.decodertf.eval()

    # By default restore middle_RNN to same model as Prior, but can restore from already trained middle_RNN too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.decodertf.load_state_dict(torch.load(restore_prior_from, map_location={'cuda:0': 'cuda:0'}))
    else:
        Prior.decodertf.load_state_dict(
            torch.load(restore_prior_from, map_location=lambda storage, loc: storage))

    Prior.decodertf.to(device)

    smile_list = []

    for i in range(n_steps):
        seqs = Prior.generate(batch_size, max_length=140, con_token_list=token_list)

        smiles = seq_to_smiles(seqs, voc)

        smile_list.extend(smiles)

        print('step: ', i)

    smile_list = pd.DataFrame(smile_list)
    smile_list.to_csv(save_file, header=False, index=False)


if __name__ == "__main__":
    max_seq_length = 140
    # num_tokens=71
    # vocab_size=71
    d_model = 128
    # num_encoder_layers = 6
    num_decoder_layers = 12
    dim_feedforward = 512
    nhead = 8
    pos_dropout = 0.1
    trans_dropout = 0.1
    n_warmup_steps = 500

    num_epochs = 600
    batch_size = 128

    n_steps = 5000

    token_list = ['is_DRD2', 'high_QED', 'good_SA']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=500)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=128)
    parser.add_argument('--prior', action='store', dest='restore_prior_from',
                        default='./data/Prior.ckpt',
                        help='Path to an c-Transformer checkpoint file to use as a Prior')

    parser.add_argument('--save_molecules_path', action='store', dest='save_file',
                        default='test.csv')

    arg_dict = vars(parser.parse_args())

    Transformer_generator(**arg_dict)
