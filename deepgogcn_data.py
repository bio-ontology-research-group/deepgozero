import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT, NAMESPACES
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_annots.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--out-terms-file', '-onf', default='data/terms.pkl',
    help='')
def main(go_file, data_file, out_terms_file):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')

    df = pd.read_pickle(data_file)
    print("DATA FILE" ,len(df))
    
    logging.info('Processing annotations')
    
    cnt = Counter()
    annotations = list()
    mf = Counter()
    bp = Counter()
    cc = Counter()
    iprs = Counter()
    for i, row in df.iterrows():
        for term in row['dg_annotations']:
            cnt[term] += 1
        for term in row['prop_annotations']:
            if go.get_namespace(term) == NAMESPACES['mf']:
                mf[term] += 1
            elif go.get_namespace(term) == NAMESPACES['bp']:
                bp[term] += 1
            else:
                cc[term] += 1
        for ipr in row.interpros:
            iprs[ipr] += 1

    iprs = list(iprs.keys())
    mf = list(mf.keys())
    bp = list(bp.keys())
    cc = list(cc.keys())
    terms = list(cnt.keys())

    logging.info(f'Number of terms {len(terms)}')
    logging.info(f'Number of iprs {len(iprs)}')
    
    # Save the list of terms
    terms_df = pd.DataFrame({'gos': terms})
    terms_df.to_pickle(out_terms_file)
    mf_df = pd.DataFrame({'gos': mf})
    mf_df.to_pickle('data/mf.pkl')
    bp_df = pd.DataFrame({'gos': bp})
    bp_df.to_pickle('data/bp.pkl')
    cc_df = pd.DataFrame({'gos': cc})
    cc_df.to_pickle('data/cc.pkl')

    ipr_df = pd.DataFrame({'interpros': iprs})
    ipr_df.to_pickle('data/interpros.pkl')


if __name__ == '__main__':
    main()
