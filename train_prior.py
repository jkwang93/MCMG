#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import decoderTransformer, transformer_RL
from torch.optim import Adam
from Optim import ScheduledOptim
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
batch_size = 1024
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

print(device)


def pretrain(restore_from=None):
    """Trains the Prior decodertf"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="./data/RE/Voc_RE1")

    # Create a Dataset from a SMILES file
    moldata = MolData("./data/RE/RE_data_filtered.smi", voc)
    # moldata = MolData("../data/RE/small.smi", voc)

    valid = MolData("./data/RE/chembl_validation.smi", voc)
    train_data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    valid_data = DataLoader(valid, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = transformer_RL(voc, d_model, nhead, num_decoder_layers,
                           dim_feedforward, max_seq_length,
                           pos_dropout, trans_dropout)

    # Can restore from a saved decodertf
    if restore_from:
        Prior.decodertf.load_state_dict(torch.load(restore_from))

    # optimizer = torch.optim.Adam(Prior.decodertf.parameters(), lr=0.001)

    optim = ScheduledOptim(
        Adam(Prior.decodertf.parameters(), betas=(0.9, 0.98), eps=1e-09),
        d_model * 8,n_warmup_steps)

    train_losses, val_losses = train(train_data, valid_data, Prior, optim, num_epochs)

    torch.cuda.empty_cache()



def train(train_data, valid_data, model, optim, num_epochs):
    model.decodertf.to(device)


    model.decodertf.train()
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    early_stopping = EarlyStopping(patience=5, verbose=False)

    for epoch in range(num_epochs):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        total_loss = 0
        for step, batch in tqdm(enumerate(train_data), total=len(train_data)):


            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule

            loss, each_molecule_loss = model.likelihood(seqs)
            # loss = - log_p.mean()

            # Calculate gradients and take a step
            optim.zero_grad()
            loss.backward()
            optim.step_and_update_lr()
            # print(loss)

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))

            # if step % print_every == print_every - 1:


            if step % 200 == 0 and step != 0:
                # decrease_learning_rate(optim, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))

        print('average epoch loss:', total_loss / len(train_data))
        val_loss = validate(valid_data, model)
        val_losses.append((total_step, val_loss))

        early_stopping(val_loss, model.decodertf, 'RE1_Prior')

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Save the Prior
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model.decodertf.state_dict(), "./output/RE1_Prior.ckpt")
        print(f'Val Loss: {val_loss}')
    return train_losses, val_losses
        # Every 500 steps we decrease learning rate and print some information
        # if step % 10 == 0 and step != 0:
        #     decrease_learning_rate(optimizer, decrease_by=0.03)
        #     # tqdm.write("*" * 50)
        #     # tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
        #     seqs, likelihood, _ = Prior.sample(128)
        #     valid = 0
        #     for i, seq in enumerate(seqs.cpu().numpy()):
        #         smile = voc.decode(seq)
        #         if Chem.MolFromSmiles(smile):
        #             valid += 1
        #         if i < 5:
        #             tqdm.write(smile)
        #     tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
        #     tqdm.write("*" * 50 + "\n")
        #     torch.save(Prior.decodertf.state_dict(), "../data/Prior.ckpt")






def validate(valid_data, model):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.decodertf.to(device)
    model.decodertf.eval()
    total_loss = 0

    for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):
        with torch.no_grad():
            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule
            loss, each_molecule_loss = model.likelihood(seqs)
            # loss = - log_p.mean()

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))
    return total_loss / len(valid_data)



if __name__ == "__main__":
    pretrain()
