#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
from matplotlib import pyplot as plt
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


ont = 'cc'

@ck.command()
@ck.option(
    '--test-data-file', '-tsdf', default=f'data-netgo/{ont}/test_data.pkl',
    help='Test data file')
@ck.option(
    '--terms-file', '-tf', default=f'data-netgo/{ont}/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--netgo-scores-file', '-tsf', default=f'data-netgo/netgo-{ont}.txt',
    help='NetGO predictions')
@ck.option(
    '--out_file', '-of', default=f'data-netgo/{ont}/predictions_netgo.pkl', help='Output file')
def main(test_data_file, terms_file,
         netgo_scores_file, out_file):

    go_rels = Ontology('data-netgo/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    test_df = pd.read_pickle(test_data_file)
    
    
    netgo_scores = {}
    with open(netgo_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if len(it) < 3:
                continue
            p_id, go_id, score = it[0], it[1], float(it[2])
            if p_id not in netgo_scores:
                netgo_scores[p_id] = {}
            netgo_scores[p_id][go_id] = score
    preds = []
    print('NetGO preds')
    
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prop_annots = {}
        prot_id = row.proteins
        if prot_id in netgo_scores:
            annots = netgo_scores[prot_id]
            prop_annots = annots.copy()
            for go_id, score in annots.items():
                for sup_go in go_rels.get_anchestors(go_id):
                    if sup_go in prop_annots:
                        prop_annots[sup_go] = max(prop_annots[sup_go], score)
                    else:
                        prop_annots[sup_go] = score
        pred_scores = np.zeros(len(terms), dtype=np.float32)
        for i, go_id in enumerate(terms):
            if go_id in prop_annots:
                pred_scores[i] = prop_annots[go_id]

        preds.append(pred_scores)

    test_df['preds'] = preds
    test_df.to_pickle(out_file)

if __name__ == '__main__':
    main()
