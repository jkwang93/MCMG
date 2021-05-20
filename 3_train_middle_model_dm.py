#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from MCMG_utils.data_structs import MolData, Vocabulary
from models.model_rnn import RNN
from MCMG_utils.utils import  decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def train_middle(train_data, save_model='./DM.ckpt'):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc_RE1")

    # Create a Dataset from a SMILES file
    moldata = MolData(train_data, voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)


    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    for epoch in range(1, 9):

        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
              
            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                print(loss.cpu().data)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.cpu().data))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), save_model)

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--train-data', action='store', dest='train_data')
    parser.add_argument('--save-middle-path', action='store', dest='save_dir',
                        help='Path and name of middle model.')

    arg_dict = vars(parser.parse_args())

    train_middle(**arg_dict)