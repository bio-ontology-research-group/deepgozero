import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT, NAMESPACES, read_fasta
import logging
import os

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data-cafa3/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data-cafa3/swissprot_exp.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--sim-file', '-sf', default='data-cafa3/swissprot_exp.sim',
    help='Sequence similarity generated with Diamond')
def main(go_file, data_file, sim_file):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')

    df = pd.read_pickle(data_file)
    proteins = set(df['proteins'].values)
    
    print("DATA FILE" ,len(df))

    print("Loading CAFA3 data")

    targets, seqs = read_fasta('data-cafa3/CAFA3_targets/targets_all.fasta')
    sequences = {t: s for t, s in zip(targets, seqs)}
    nk_proteins = set()
    nk_files = ['bpo_all_type1.txt', 'mfo_all_type1.txt', 'cco_all_type1.txt']
    for filename in nk_files:
        with open(f'data-cafa3/benchmark20171115/lists/{filename}') as f:
            proteins = f.read().splitlines()
            nk_proteins |= set(proteins)
    exp_annots = {}
    with open('data-cafa3/benchmark20171115/groundtruth/leafonly_all.txt') as f:
        for line in f:
            target_id, go_id = line.strip().split('\t')
            if target_id not in nk_proteins:
                continue
            if target_id not in exp_annots:
                exp_annots[target_id] = []
            exp_annots[target_id].append(go_id)

    interpros = {}
    with open('data-cafa3/benchmark20171115/targets.fasta.tsv') as f:
        for line in f:
            it = line.strip().split('\t')
            t_id, ipr = it[0], it[11]
            if t_id not in interpros:
                interpros[t_id] = set()
            interpros[t_id].add(ipr)
    
    seqs = []
    targets = []
    exp_annotations = []
    prop_annotations = []
    iprs = []
    for t_id, annots in exp_annots.items():
        targets.append(t_id)
        exp_annotations.append(annots)
        seqs.append(sequences[t_id])
        annot_set = set()
        for go_id in annots:
            annot_set |= go.get_anchestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
        if t_id in interpros:
            iprs.append(interpros[t_id])
        else:
            iprs.append(set())
    test_df = pd.DataFrame({
        'proteins': targets,
        'sequences': seqs,
        'exp_annotations': exp_annotations,
        'prop_annotations': prop_annotations,
        'interpros': iprs
    })
        
    print('Processing train and valid annotations')
    
    annotations = list()
    for ont in ['mf', 'bp', 'cc']:
        cnt = Counter()
        iprs = Counter()
        index = []
        test_index = []
        for i, row in enumerate(df.itertuples()):
            ok = False
            for term in row.prop_annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt[term] += 1
                    ok = True
            for ipr in row.interpros:
                iprs[ipr] += 1
            if ok:
                index.append(i)

        for i, row in enumerate(test_df.itertuples()):
            ok = False
            for term in row.prop_annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    ok = True
                if len(row.interpros) == 0:
                    ok = False
            if ok:
                test_index.append(i)

            
        del cnt[FUNC_DICT[ont]] # Remove top term
        tdf = df.iloc[index]
        terms = list(cnt.keys())
        interpros = list(iprs.keys())

        print(f'Number of {ont} terms {len(terms)}')
        print(f'Number of {ont} iprs {len(iprs)}')
        print(f'Number of {ont} proteins {len(tdf)}')
    
        terms_df = pd.DataFrame({'gos': terms})
        terms_df.to_pickle(f'data-cafa3/{ont}/terms.pkl')
        iprs_df = pd.DataFrame({'interpros': interpros})
        iprs_df.to_pickle(f'data-cafa3/{ont}/interpros.pkl')

        # Split train/valid/test
        proteins = tdf['proteins']
        prot_set = set(proteins)
        prot_idx = {v:k for k, v in enumerate(proteins)}
        sim = {}
        train_prots = set()
        with open(sim_file) as f:
            for line in f:
                it = line.strip().split('\t')
                p1, p2, score = it[0], it[1], float(it[2]) / 100.0
                if p1 == p2:
                    continue
                if score < 0.5:
                    continue
                if p1 not in prot_set or p2 not in prot_set:
                    continue
                if p1 not in sim:
                    sim[p1] = []
                if p2 not in sim:
                    sim[p2] = []
                sim[p1].append(p2)
                sim[p2].append(p1)

        used = set()
        def dfs(prots, prot):
            used.add(prot)
            if prot in sim:
                for p in sim[prot]:
                    if p not in used:
                        dfs(prots, p)
            prots.append(prot)

        groups = []
        for p in proteins:
            group = []
            if p not in used:
                dfs(group, p)
                groups.append(group)
        print(len(proteins), len(groups))
        index = np.arange(len(groups))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        train_n = int(len(groups) * 0.9)
        train_index = []
        valid_index = []
        for idx in index[:train_n]:
            for prot in groups[idx]:
                train_index.append(prot_idx[prot])
        for idx in index[train_n:]:
            for prot in groups[idx]:
                valid_index.append(prot_idx[prot])
                
        train_index = np.array(train_index)
        valid_index = np.array(valid_index)
        
        train_df = tdf.iloc[train_index]
        train_df.to_pickle(f'data-cafa3/{ont}/train_data.pkl')
        valid_df = tdf.iloc[valid_index]
        valid_df.to_pickle(f'data-cafa3/{ont}/valid_data.pkl')
        ts_df = test_df.iloc[test_index]
        ts_df.to_pickle(f'data-cafa3/{ont}/test_data.pkl')
        
        
        print(f'Train/Valid proteins for {ont} {len(train_df)}/{len(valid_df)}')
        print(f'Test proteins for {ont} {len(ts_df)}')


if __name__ == '__main__':
    main()
