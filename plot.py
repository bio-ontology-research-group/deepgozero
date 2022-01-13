from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import click as ck
from sklearn.metrics import auc
from utils import Ontology

@ck.command()
def main():
    onts = ['mf', 'bp', 'cc']

    go = Ontology('data/go.obo', with_rels=True)
    
    eval_terms = {
        'mf': ['GO:0001227', 'GO:0001228', 'GO:0003735', 'GO:0004867', 'GO:0005096'],
        'bp': ['GO:0000381', 'GO:0032729', 'GO:0032755', 'GO:0032760', 'GO:0046330', 'GO:0051897', 'GO:0120162'],
        'cc': ['GO:0005762', 'GO:0022625', 'GO:0042788', 'GO:1904813']}

    # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.00])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Zero-shot prediction performance')
    # plt.tight_layout()
    # ax = plt.gca() 
    # ax.set_aspect('equal')
    avgs = [0, 0, 0]
    for i, ont in enumerate(onts):
        for term in eval_terms[ont]:
            aucs = []
            df = pd.read_pickle(f'data/{ont}/zero_{term}_auc.pkl')
            fpr, tpr = df['fpr'], df['tpr']
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, label=f'{term}, AUC = {roc_auc:.3f}')
            # plt.legend(loc=4)
            df = pd.read_pickle(f'data/{ont}/zero_{term}_auc_all.pkl')
            fpr, tpr = df['fpr'], df['tpr']
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, label=f'{term} (all), AUC = {roc_auc:.3f}')
            # plt.legend(loc=4)
            df = pd.read_pickle(f'data/{ont}/zero_{term}_auc_train.pkl')
            fpr, tpr = df['fpr'], df['tpr']
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, label=f'{term}*, AUC = {roc_auc:.3f}')
            # plt.legend(loc=4)
            name = go.get_term(term)['name']
            print('\\hline')
            print(f'{ont} & {term} & {name} & {aucs[0]:.3f} & {aucs[1]:.3f} & {aucs[2]:.3f} \\\\')
            for i, value in enumerate(aucs):
                avgs[i] += value
    print('\\hline')
    print(f' & & Average & {avgs[0] / 16:.3f} & {avgs[1] / 16:.3f} & {avgs[2] / 16:.3f} \\\\')
    print('\\hline')
    # plt.savefig('data/zeroshot.eps')

if __name__ == '__main__':
    main()
