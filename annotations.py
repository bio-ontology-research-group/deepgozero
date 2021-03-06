#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_2021_03.pkl',
    help='Pandas dataframe with protein sequences')
@ck.option(
    '--out-file', '-o', default='data/swissprot_exp_annots_2021_03.tab',
    help='Fasta file')
def main(data_file, out_file):
    # Load interpro data
    df = pd.read_pickle(data_file)
    print(len(df)) 
    with open(out_file, 'w') as f:
        for row in df.itertuples():
            f.write(row.proteins)
            for go_id in row.prop_annotations:
                f.write('\t' + go_id.replace(':', '_'))
            f.write('\n')
    

if __name__ == '__main__':
    main()
