#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from torch.optim import Adam
from Optim import ScheduledOptim
from model_rnn import RNN
from save.pytorchtools import EarlyStopping
from utils import Variable, decrease_learning_rate

# rdBase.DisableLog('rdApp.error')


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
batch_size = 100
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

print(device)


def pretrain(restore_from='Agent.ckpt'):
    # token_list = ['is_JNK3', 'is_GSK3', 'high_QED', 'good_SA']

    smile_list = []
    """Trains the Prior rnn"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="./data/RE/Voc_RE1")

    Prior = RNN(voc)

    # Can restore from a saved rnn
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_from))

    else:
        Prior.rnn.load_state_dict(torch.load(restore_from, map_location=lambda storage, loc: storage))

    Prior.rnn.to(device)



    Prior.rnn.eval()
    valid = 0
    #
    # seqs, likelihood, _ = Prior.sample(128)
    # valid = 0
    # for i, seq in enumerate(seqs.cpu().numpy()):
    #     smile = voc.decode(seq)
    #     if Chem.MolFromSmiles(smile):
    #         valid += 1
    #     if i < 5:
    #         tqdm.write(smile)

    for step in range(50):
        # Every 500 steps we decrease learning rate and print some information
        # seqs = Prior.generate(batch_size, max_length=140, con_token_list=token_list)
        seqs, likelihood, _ = Prior.sample(batch_size)


        for i, seq in enumerate(seqs.cpu().numpy()):
            smile = voc.decode(seq)
            mol = Chem.MolFromSmiles(smile)

            if mol != None:
                valid += 1
            else:
                print(smile)
            # if i < 5:
            #     tqdm.write(smile)
            smile_list.append(smile)
    print(valid / (batch_size * 50))
    write_in_file('./output/4_condition/no_one_hot/4_100_Agent.smi', smile_list)
    #
    # tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
    # tqdm.write("*" * 50 + "\n")
    # torch.save(Prior.rnn.state_dict(), "../data/Prior.ckpt")


def write_in_file(path_to_file, data):
    with open(path_to_file, 'a+') as f:
        for item in data:
            f.write("%s\n" % item)


if __name__ == "__main__":
    pretrain()
    torch.cuda.empty_cache()
