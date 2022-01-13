#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os
import torch as th

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging
import json

from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from utils import get_goplus_defs, MOLECULAR_FUNCTION, CELLULAR_COMPONENT, BIOLOGICAL_PROCESS

logging.basicConfig(level=logging.INFO)

from deepgoel import DGELModel, load_normal_forms

ont = 'cc'

# eval_terms = {
#     'mf': ['GO:0001227', 'GO:0001228', 'GO:0003735', 'GO:0004867', 'GO:0005096'],
#     'bp': ['GO:0000381', 'GO:0032729', 'GO:0032755', 'GO:0032760', 'GO:0046330', 'GO:0051897', 'GO:0120162'],
#     'cc': ['GO:0005762', 'GO:0022625', 'GO:0042788', 'GO:1904813']}

@ck.command()
@ck.option(
    '--train-data-file', '-tsdf', default=f'data/{ont}/train_data.pkl',
    help='Test data file')
@ck.option(
    '--valid-data-file', '-tsdf', default=f'data/{ont}/valid_data.pkl',
    help='Test data file')
@ck.option(
    '--test-data-file', '-tsdf', default=f'data/{ont}/test_data.pkl',
    help='Test data file')
@ck.option(
    '--terms-file', '-tf', default=f'data/{ont}/terms_zero_10.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--model-file', '-mf', default=f'data/{ont}/deepgozero_zero_10.th',
    help='Prediction model')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(train_data_file, valid_data_file, test_data_file, terms_file, model_file, device):
    # Load interpro data
    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    test_df = pd.read_pickle(test_data_file)
    df = pd.concat([train_df, valid_df, test_df])
    # df = test_df
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    
    ipr_df = pd.read_pickle(f'data/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    nf1, nf2, nf3, nf4, rels_dict, zero_classes = load_normal_forms(
        'data/go.norm', terms_dict)

    defins = get_goplus_defs('data/definitions_go.txt')
    with open('data/eval_terms.json') as f:
        eval_terms = json.loads(f.read())
    zero_terms = [term for term in eval_terms[ont] if term in zero_classes ] #and term in defins]
    print(len(zero_terms))
    
    net = DGELModel(len(iprs_dict), len(terms), len(zero_classes), len(rels_dict), device).to(device)
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()

    zero_terms_dict = {v: k for k, v in enumerate(zero_terms)}
    data, labels = get_data(df, iprs_dict, zero_terms_dict)
    data = data.to(device)
    labels = labels.detach().cpu().numpy()

    go_data = th.zeros(len(zero_terms), dtype=th.long).to(device)
    for i, term in enumerate(zero_terms):
        go_data[i] = zero_classes[term]
    zero_score = net.predict_zero(data, go_data).cpu().detach().numpy()
    total = 0
    for i, term in enumerate(zero_terms):
        roc_auc, fpr, tpr = compute_roc(labels[:, i], zero_score[:, i])
        print(term, roc_auc)
        total += roc_auc
    # print(f'Average {total / len(zero_terms):.3f}')
        # df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        # df.to_pickle(f'data/{ont}/zero_{term}_auc_all.pkl')
        
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def compute_fmax(labels, preds):
    fmax = 0.0
    pmax = 0
    rmax = 0
    patience = 0
    precs = []
    recs = []
    for t in range(0, 101):
        threshold = t / 100.0
        predictions = (preds >= threshold).astype(np.float32)
        tp = np.sum(labels * predictions, axis=1)
        fp = np.sum(predictions, axis=1) - tp
        fn = np.sum(labels, axis=1) - tp
        tp_ind = tp > 0
        tp = tp[tp_ind]
        fp = fp[tp_ind]
        fn = fn[tp_ind]
        if len(tp) == 0:
            continue
        p = np.mean(tp / (tp + fp))
        r = np.sum(tp / (tp + fn)) / len(tp_ind)
        precs.append(p)
        recs.append(r)
        f = 2 * p * r / (p + r)
        if fmax <= f:
            fmax = f
    return fmax, precs, recs


def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
