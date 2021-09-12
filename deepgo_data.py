import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-ndf', default='data/swissprot_interactions.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--out-terms-file', '-onf', default='data/terms.pkl',
    help='')
@ck.option(
    '--train-data-file', '-trdf', default='data/train_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--test-data-file', '-tsdf', default='data/test_data.pkl',
    help='Result file with a list of terms for prediction task')
def main(go_file, data_file,
         out_terms_file, train_data_file, test_data_file):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')

    df = pd.read_pickle(data_file)
    print("DATA FILE" ,len(df))
    
    logging.info('Processing annotations')
    
    cnt = Counter()
    annotations = list()
    for i, row in df.iterrows():
        for term in row['prop_annotations']:
            cnt[term] += 1
    
    terms = list(cnt.keys())

    logging.info(f'Number of terms {len(terms)}')
    
    # Save the list of terms
    terms_df = pd.DataFrame({'gos': terms})
    terms_df.to_pickle(out_terms_file)

    n = len(df)
    # Split train/valid
    index = np.arange(n)
    train_n = int(n * 0.95)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    test_df = df.iloc[index[train_n:]]

    print('Number of train proteins', len(train_df))
    train_df.to_pickle(train_data_file)

    print('Number of test proteins', len(test_df))
    test_df.to_pickle(test_data_file)


if __name__ == '__main__':
    main()
