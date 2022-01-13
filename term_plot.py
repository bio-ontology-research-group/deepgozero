from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import click as ck
from sklearn.metrics import auc
from utils import Ontology
from collections import Counter

@ck.command()
def main():
    onts = ['mf', 'bp', 'cc']

    go = Ontology('data/go.obo', with_rels=True)
    

    plt.figure()
#    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('Annotations')
    plt.ylabel('Performance, AUC')
    plt.title('Term centric performance by annotation number')
    avgs = [0, 0, 0]
    all_aucs = []
    all_annots = []
    mlp_aucs = []
    mlp_annots = []
    
    for i, ont in enumerate(onts):
        df = pd.read_pickle(f'data/{ont}/deepgozero_auc_annots.pkl')
        annots, aucs = df['annots'], df['aucs']
        all_annots += list(annots)
        all_aucs += list(aucs)
        df = pd.read_pickle(f'data/{ont}/mlp_auc_annots.pkl')
        annots, aucs = df['annots'], df['aucs']
        mlp_annots += list(annots)
        mlp_aucs += list(aucs)

    all_annots, all_aucs, all_err = get_average(all_annots, all_aucs)
    plt.errorbar(all_annots, all_aucs, yerr=all_err, fmt = 'o', label=f'DeepGOZero')
    plt.legend(loc=4)
    mlp_annots, mlp_aucs, mlp_err = get_average(mlp_annots, mlp_aucs)
    plt.errorbar(mlp_annots, mlp_aucs, yerr=mlp_err, fmt = 'o', label=f'MLP')
    plt.legend(loc=4)
    plt.gcf().autofmt_xdate()
    plt.savefig('data/annot-terms.eps')

def get_average(annots, aucs):
    total = Counter()
    avg = {}
    err = {}
    for ann, auc in zip(annots, aucs):
        ann //= 5
        if ann not in avg:
            avg[ann] = []
        avg[ann].append(auc)
    res_ann = []
    res_auc = []
    res_err = []
    for i in range(0, 20):
        if i in avg:
            a = np.array(avg[i])
            res_auc.append(a.mean())
            res_err.append(np.absolute(a - a.mean()).mean())
            res_ann.append(f'{i * 5 + 1}-{(i + 1) * 5}')
    return res_ann, res_auc, res_err

if __name__ == '__main__':
    main()
