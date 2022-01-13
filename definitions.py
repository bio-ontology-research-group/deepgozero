#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging
from utils import Ontology, get_goplus_defs, NAMESPACES

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/definitions_go.txt',
    help='Pandas dataframe with protein sequences')
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology File')
@ck.option(
    '--sub-ontology', '-so', default='bp',
    help='Sub ontology to filter definitions')
@ck.option(
    '--out-file', '-of', default='data/definitions_go_bp.txt',
    help='Output file')
def main(data_file, go_file, sub_ontology, out_file):
    go = Ontology(go_file)
    gos = go.get_namespace_terms(NAMESPACES[sub_ontology])
    out = open(out_file, 'w')                     
    with open(data_file) as f:
        for line in f:
            go_id = line[:10].replace('_', ':')
            if go_id in gos:
                out.write(line)
    out.close()
    


if __name__ == '__main__':
    main()
