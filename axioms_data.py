#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl',
    help='Pandas dataframe with protein sequences')
@ck.option(
    '--annots-file', '-af', default='data/swissprot_annots_new.tab',
    help='Annotations file produced by Axioms.groovy')
def main(data_file, annots_file):
    # Load interpro data
    df = pd.read_pickle(data_file)

    dg_annots = []
    with open(annots_file) as f:
        for line in f:
            it = line.strip().split('\t')
            dg_annots.append(it[1:])
    df['dg_annotations'] = dg_annots
    name, ext = os.path.splitext(data_file)
    df.to_pickle(name + '_annots.pkl')


if __name__ == '__main__':
    main()
