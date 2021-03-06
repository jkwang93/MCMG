import sys
import rdkit
from argparse import ArgumentParser
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import scripts.sascorer as sascorer
import rdkit.Chem.QED as QED
import pandas as pd
import seaborn as sns
from scipy import stats, integrate
from matplotlib import pyplot as plt

# parser = ArgumentParser()
# parser.add_argument('--ref_path', required=True)
# args = parser.parse_args()
#
# lg = rdkit.RDLogger.logger()
# lg.setLevel(rdkit.RDLogger.CRITICAL)

# pred_data = [line.split()[1:] for line in sys.stdin]
# pred_mols = [mol for mol,x,y,qed,sa in pred_data if float(x) >= 0.5 and float(y) >= 0.5 and float(qed) > 0.6 and float(sa) < 4]
#
# fraction_actives = len(pred_mols) / len(pred_data)
# print('fraction actives:', fraction_actives)

# pred_mols = pd.read_csv('../output/results/activate_molecule_csv/one_hot_50_RL_positive.csv',
#                         header=None).values.reshape(-1)
pred_mols = pd.read_csv('../output/results/activate_molecule_csv/RE_agent_no_1500steps_positive.csv',
                        header=None).values.reshape(-1)
# pred_mols = pd.read_csv('2_con_transformer.smi',header=None).values.reshape(-1)

ref_path = 'actives.txt'

with open(ref_path) as f:
    next(f)
    true_mols = [line.split(',')[0] for line in f]
print('number of active reference', len(true_mols))

true_mols = [Chem.MolFromSmiles(s) for s in true_mols]
true_mols = [x for x in true_mols if x is not None]
true_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in true_mols]

pred_mols = [Chem.MolFromSmiles(s) for s in pred_mols]
pred_mols = [x for x in pred_mols if x is not None]
pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]

fraction_similar = 0

sim_distribution = []
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)

    if max(sims) >= 0.4:
        fraction_similar += 1
    sim_distribution.append(max(sims))

print('novelty:', 1 - fraction_similar / len(pred_mols))

similarity = 0
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
    similarity += sum(sims)

n = len(pred_fps)
n_pairs = n * (n - 1) / 2
diversity = 1 - similarity / n_pairs
print('diversity:', diversity)

# plot
print(len(sim_distribution))
# sns.distplot(sim_distribution,fit=stats.gamma)
# sns.kdeplot(sim_distribution, shade=True)

# kernels = ['gau', 'cos', 'biw', 'epa', 'tri', 'triw']
# for k in kernels:
#     sns.kdeplot(sim_distribution, kernel=k, label=k)

# sns.kdeplot(sim_distribution, bw=0.05,shade=True)
#
#
# plt.show()
