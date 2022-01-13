#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import NAMESPACES, Ontology, get_goplus_defs
from collections import Counter
import json
from deepgoel import load_normal_forms

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl',
    help='DataFrame with proteins, sequences and annotations')
@ck.option(
    '--out-file', '-of', default='data/swissprot_exp_zero_10.pkl',
    help='With annotations for zero shot prediction')
def main(data_file, out_file):
    go = Ontology('data/go.obo', with_rels=True)

    defins = get_goplus_defs('data/definitions_go.txt')
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        'data/go.norm', {})

    annots = Counter()
    df = pd.read_pickle(data_file)
    for i, row in df.iterrows():
        annots.update(row.prop_annotations)
    spec_terms = []
    eval_terms = {}
    annots_sum = 0
    for go_id in go.ont:
        # gid = go_id.replace(':', '_')
        if annots[go_id] > 0 and annots[go_id] < 10:
            ns = go.get_namespace(go_id).split('_')
            ns = ns[0][0] + ns[1][0]
            if go_id not in zero_classes:
                continue
            print(go_id, ns, annots[go_id], go.get_term(go_id)['name'], go_id in defins)
            annots_sum += annots[go_id]
            spec_terms.append(go_id)
            if ns not in eval_terms:
                eval_terms[ns] = []
            eval_terms[ns].append(go_id)
        elif annots[go_id] == 0 and go_id in zero_classes:
            print(go_id, ns, 0, go.get_term(go_id)['name'], go_id in defins)
    return
    with open('data/eval_terms.json', 'w') as f:
        f.write(json.dumps(eval_terms))
    df = pd.read_pickle(data_file)
    zeros = set(spec_terms)
    zero_annotations = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']
        for go_id in annots:
            if go_id not in zeros:
                annot_set |= go.get_anchestors(go_id)
        annots = list(annot_set)
        zero_annotations.append(annots)
    df['zero_annotations'] = zero_annotations
    df.to_pickle(out_file)

if __name__ == '__main__':
    main()
