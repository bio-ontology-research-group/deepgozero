import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT, NAMESPACES
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_interactions.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--out-terms-file', '-onf', default='data/interpros.pkl',
    help='')
@ck.option(
    '--train-data-file', '-trdf', default='data/interpro_train_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--test-data-file', '-tsdf', default='data/interpro_test_data.pkl',
    help='Result file with a list of terms for prediction task')
def main(data_file,
         out_terms_file, train_data_file, test_data_file):
    df = pd.read_pickle(data_file)
    print("DATA FILE" ,len(df))
    
    logging.info('Processing annotations')

    annots = {}
    
    with open('data/interproscan.tsv') as f:
        for line in f:
            it = line.strip().split('\t')
            p_id = it[0]
            if p_id not in annots:
                annots[p_id] = set()
            annots[p_id].add(it[4])
    
    iprs = Counter()
    index = []
    for p_id, annot in annots.items():
        for ipr in annot:
            iprs[ipr] += 1
    # Clean up
    for ipr in list(iprs.keys()):
        if iprs[ipr] < 3:
            del iprs[ipr]
    print(len(iprs))
    print(iprs.most_common())

    annotations = []
    for i, row in enumerate(df.itertuples()):
        if row.proteins not in annots:
            continue
        index.append(i)
        annotations.append(annots[row.proteins])
    df = df.iloc[index]
    df = df.reset_index()
    df['interproscan'] = annotations
    iprs = list(iprs.keys())
    logging.info(f'Number of iprs {len(iprs)}')
    
    # Save the list of terms
    ipr_df = pd.DataFrame({'interpros': iprs})
    ipr_df.to_pickle('data/interpros.pkl')

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
