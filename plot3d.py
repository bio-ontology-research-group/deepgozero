#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque
from mpl_toolkits.mplot3d import Axes3D

from utils import Ontology, FUNC_DICT

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
from deepgoel import DGELModel, load_normal_forms
import torch as th

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
def main(data_root, ont):
    go_file = f'{data_root}/go.norm'
    model_file = f'{data_root}/{ont}/deepgozero.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'

    device = 'cpu:0'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)

    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)
    
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    
    normal_forms = nf1, nf2, nf3, nf4
    nf1 = th.LongTensor(nf1).to(device)
    nf2 = th.LongTensor(nf2).to(device)
    nf3 = th.LongTensor(nf3).to(device)
    nf4 = th.LongTensor(nf4).to(device)
    normal_forms = nf1, nf2, nf3, nf4

    net = DGELModel(n_iprs, n_terms, n_zeros, n_rels, device).to(device)
    print(net)
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))

    plot_classes = ['GO:0003674', 'GO:0005488', 'GO:0003824', ]
    
    classes = [t_id for t_id in plot_classes if t_id in terms_dict] 
    plot_ids = [terms_dict[t_id] for t_id in classes] 

    embeds = net.go_embed(th.LongTensor(plot_ids)).detach().numpy()
    rs = np.abs(net.go_rad(th.LongTensor(plot_ids)).detach().numpy())
    
    plot_embeddings(embeds, rs, classes)

def plot_embeddings(embeds, rs, classes):
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    embeds = TSNE().fit_transform(embeds)
    print(embeds, rs)
    fig =  plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    
    
    for i in range(embeds.shape[0]):
        a, b = embeds[i, 0], embeds[i, 1]
        r = rs[i]
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v)) + a
        y = r * np.outer(np.sin(u), np.sin(v)) + b
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=colors[(i + 2) % len(colors)], rstride=4, cstride=4, linewidth=0, alpha=0.3)
        # ax.annotate(classes[i][1:-1], xy=(x, y + r + 0.03), fontsize=10, ha="center", color=colors[i % len(colors)])
    filename = 'embeds3d.pdf'
    plt.savefig(filename)
    # plt.show()

    
if __name__ == '__main__':
    main()
