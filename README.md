# MCMG
MCMG: multi-constraints molecular generation approach based on conditional transformer and reinforcement learning

## Environment
- python = 3.8.3
- pytroch = 1.6.0
- RDKit
- numpy
- pandas



## How to run？
The default task of our code is the DRD2 target. Users can customize their own tasks, which need to be modified in data_structs in MCMG_utils and 1-4 training code.

- There is an example of initial data: it is in data/drd2/con_RE_filter_test_drd.csv

```
python 1_train_prior_Transformer.py --train-data {your_training_data_path} --valid-data {your_valid_data_path} --save-prior-path {path_to_save_prior_model}

python 2_generator_Transformer.py --prior {piror_model_path} --save_molecules_path {save_molecules_path}

python 3_train_middle_model_dm.py --train-data {your_training_data_path} --save-middle-path {path_to_save_middle_model}

python 4_train_agent_save_smiles.py  --num-steps 5000 --batch-size 128 --middle {path_of_middle_model} --agent {path_to_save_agent_model} ---save-file-path{save_smiles}
```