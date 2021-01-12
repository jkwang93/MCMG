#!/usr/bin/env python

import torch

import time


from model import transformer_RL
from data_structs import Vocabulary, Experience
from model_rnn import RNN
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, decrease_learning_rate
import pandas as pd


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

n_steps = 2000

# token_list = ['is_JNK3', 'is_GSK3', 'high_QED', 'good_SA']
# token_list = ['is_JNK3', 'is_GSK3']
token_list = ['high_QED', 'good_SA']



device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')



def train_RNN(restore_prior_from='output/Prior.ckpt',
              restore_RNN_from='./output/middle_RNN.ckpt',
              scoring_function='tanimoto',
              scoring_function_kwargs=None,
              save_dir=None, learning_rate=0.0005,
              batch_size=batch_size, n_steps=n_steps,
              num_processes=0, sigma=60,
              experience_replay=0):
    voc = Vocabulary(init_from_file="./data/RE/Voc_RE1")

    start_time = time.time()

    Prior = transformer_RL(voc, d_model, nhead, num_decoder_layers,
                           dim_feedforward, max_seq_length,
                           pos_dropout, trans_dropout)

    # middle_RNN = RNN(voc)

    Prior.decodertf.eval()


    # logger = VizardLog('../data/logs')

    # By default restore middle_RNN to same model as Prior, but can restore from already trained middle_RNN too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.decodertf.load_state_dict(torch.load('./C2_prior/output/2c_qed_Prior.ckpt',map_location={'cuda:3':'cuda:7'}))
        # middle_RNN.rnn.load_state_dict(torch.load(restore_RNN_from))
    else:
        Prior.decodertf.load_state_dict(
            torch.load('./C2_prior/output/2c_qed_Prior.ckpt', map_location=lambda storage, loc: storage))
        # middle_RNN.rnn.load_state_dict(torch.load(restore_RNN_from, map_location=lambda storage, loc: storage))

    Prior.decodertf.to(device)


    print("Model initialized, starting training...")
    # lowest_val = 1e9

    # for step in range(n_steps):
    #
    #     # Sample from middle_RNN
    #     seqs, prior_likelihood = Prior.sample(batch_size, max_length=140, con_token_list=token_list)
    #
    #     # Remove duplicates, ie only consider unique seqs
    #     unique_idxs = unique(seqs)
    #     seqs = seqs[unique_idxs]
    #     smiles = seq_to_smiles(seqs, voc)
    smile_list = []

    for _ in range(n_steps):
        seqs = Prior.generate(500, max_length=140, con_token_list=token_list)

        smiles = seq_to_smiles(seqs, voc)

        smile_list.extend(smiles)

    smile_list = pd.DataFrame(smile_list)
    smile_list.to_csv('2C_qed_transformer_prior.csv',header=False,index=False)


if __name__ == "__main__":
    train_RNN()
    torch.cuda.empty_cache()

