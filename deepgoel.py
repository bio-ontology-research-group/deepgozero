import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from dgl.nn import GraphConv
import dgl
from torch_utils import FastTensorDataLoader

@ck.command()
@ck.option(
    '--go-file', '-hf', default='data/go.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_annots.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--model-file', '-mf', default='data/deepgoel_mf.th',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=50000,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=8,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--out-file', '-of', default='data/predictions_deepgoel.pkl',
    help='Prediction model')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(go_file, data_file, terms_file,  model_file, batch_size, epochs, load, out_file, device):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    df = pd.read_pickle(data_file)

    ipr_df = pd.read_pickle('data/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}


    loss_func = nn.BCELoss()
    features, train_data, valid_data, test_data, test_df = load_data(data_file, terms_dict, iprs_dict)
    net = DGELModel(len(iprs_dict), len(terms)).to(device)
    train_prots, train_gos, train_labels = train_data
    valid_prots, valid_gos, valid_labels = valid_data
    test_prots, test_gos, test_labels = test_data
    # train_data, train_labels = train_data.to(device), train_labels.to(device)
    # valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)
    # test_data, test_labels = test_data.to(device), test_labels.to(device)
    features = features.to(device)
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_prots) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_prots, batch_gos, batch_labels in train_loader:
                    bar.update(1)
                    batch_prots = batch_prots.to(device)
                    batch_gos = batch_gos.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(
                        features[batch_prots],
                        batch_gos)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    
            train_loss /= train_steps

            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_prots) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_prots, batch_gos, batch_labels in valid_loader:
                        bar.update(1)
                        batch_prots = batch_prots.to(device)
                        batch_gos = batch_gos.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(
                            features[batch_prots],
                            batch_gos)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                fmax = compute_fmax(valid_labels.reshape(-1, len(terms)), preds.reshape(-1, len(terms)))
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}, Fmax - {fmax}')

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model')
                    th.save(net.state_dict(), model_file)

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_prots) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_prots, batch_gos, batch_labels in test_loader:
                bar.update(1)
                batch_prots = batch_prots.to(device)
                batch_gos = batch_gos.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(
                    features[batch_prots],
                    batch_gos)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = preds.reshape(-1, len(terms))
        roc_auc = compute_roc(test_labels, preds)
        fmax = compute_fmax(test_labels.reshape(-1, len(terms)), preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}, Fmax - {fmax}')

    test_df['preds'] = list(preds)

    test_df.to_pickle(out_file)

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_fmax(labels, preds):
    fmax = 0.0
    patience = 0
    for t in range(1, 50):
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
        f = 2 * p * r / (p + r)
        if fmax <= f:
            fmax = f
            patience = 0
        else:
            patience += 1
            if patience > 10:
                return fmax
    return fmax
        

class DGELModel(nn.Module):

    def __init__(self, nb_iprs, nb_gos, embed_dim=1024):
        super().__init__()
        self.go_embed = nn.Embedding(nb_gos, embed_dim)
        self.go_bias = nn.Embedding(nb_gos, 1)
        self.fc = nn.Linear(nb_iprs, embed_dim)
        
        
    def forward(self, features, gos):
        x = th.relu(self.fc(features))
        embed = self.go_embed(gos)
        gos_bias = self.go_bias(gos)
        x = th.sum(x * embed, 1).view(-1, 1) + gos_bias
        return th.sigmoid(x)


    
    
def load_data(data_file, terms_dict, iprs_dict, fold=1):
    
    df = pd.read_pickle(data_file)
    proteins = df['proteins']
    prot_idx = {v: k for k, v in enumerate(proteins)}
    n = len(df)
    
    features = np.zeros((n, len(iprs_dict)), dtype=np.float32)
    # Filter proteins with annotations
    for i, row in enumerate(df.itertuples()):
        # seq = row.sequences
        # seq = to_onehot(seq)
        # features[i, :, :] = seq
        for ipr in row.interpros:
            features[i, iprs_dict[ipr]] = 1

    features = th.FloatTensor(features)
    

    index = np.arange(len(df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(len(index) * 0.95)
    valid_n = int(train_n * 0.95)
    train_index = index[:valid_n]
    valid_index = index[valid_n:train_n]
    test_index = index[train_n:]

    train_df = df.iloc[train_index]
    train_df.to_pickle('data/train_data.pkl')
    valid_df = df.iloc[valid_index]
    valid_df.to_pickle('data/valid_data.pkl')
    test_df = df.iloc[test_index]
    test_df.to_pickle('data/test_data.pkl')

    train_data = get_data(train_df, prot_idx, terms_dict)
    valid_data = get_data(valid_df, prot_idx, terms_dict)
    test_data = get_data(test_df, prot_idx, terms_dict)
    return features, train_data, valid_data, test_data, test_df

def get_data(df, prot_idx, terms_dict):
    proteins = th.zeros((len(df) * len(terms_dict),), dtype=th.long)
    gos = th.zeros((len(df) * len(terms_dict),), dtype=th.long)
    labels = th.zeros((len(df) * len(terms_dict), 1), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        p_id = prot_idx[row.proteins]
        for go_id in row.dg_annotations:
            g_id = terms_dict[go_id]
            proteins[i * len(terms_dict) + g_id] = p_id
            labels[i * len(terms_dict) + g_id, 0] = 1
        gos[i * len(terms_dict): (i + 1) * len(terms_dict)] = th.arange(len(terms_dict))
    return proteins, gos, labels

if __name__ == '__main__':
    main()
