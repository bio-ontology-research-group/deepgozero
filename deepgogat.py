import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from dgl.nn import GraphConv, GATConv
import dgl
from torch_utils import FastTensorDataLoader

@ck.command()
@ck.option(
    '--go-file', '-hf', default='data/go.normalized',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_annots.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/mf.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--model-file', '-mf', default='data/deepgogat_mf.th',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=67,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--out-file', '-of', default='data/predictions_deepgogat_mf.pkl',
    help='Prediction model')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(go_file, data_file, terms_file,  model_file, batch_size, epochs, load, out_file, device):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    df = pd.read_pickle(data_file)

    ipr_df = pd.read_pickle('data/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}


    loss_func = nn.BCELoss()
    train_data, valid_data, test_data, test_df, time_test_data, tdf = load_data(
        data_file, terms_dict, iprs_dict)
    go_graph = load_go_graph(terms_dict).to(device)
    
    net = DGGATModel(go_graph, len(iprs_dict), len(terms), device).to(device)
    print(net)
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    time_features, time_labels = time_test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)
    time_loader = FastTensorDataLoader(
        *time_test_data, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    time_labels = time_labels.detach().cpu().numpy()
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        # net.load_state_dict(th.load('data/deepgoel_init.th'))
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_elloss = 0
            lmbda = 0.1
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    total_loss = loss # + el_loss
                    train_loss += loss.detach().item()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                fmax = compute_fmax(valid_labels.reshape(-1, len(terms)), preds.reshape(-1, len(terms)))
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}, Fmax - {fmax}')

            print('EL Loss', train_elloss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()
            

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = preds.reshape(-1, len(terms))
        roc_auc = compute_roc(test_labels, preds)
        fmax = compute_fmax(test_labels.reshape(-1, len(terms)), preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}, Fmax - {fmax}')

    test_df['preds'] = list(preds)

    test_df.to_pickle(out_file)

    with th.no_grad():
        time_steps = int(math.ceil(len(time_labels) / batch_size))
        time_loss = 0
        preds = []
        with ck.progressbar(length=time_steps, show_pos=True) as bar:
            for batch_features, batch_labels in time_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                time_loss += batch_loss.detach().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            time_loss /= time_steps
        preds = preds.reshape(-1, len(terms))
        roc_auc = compute_roc(time_labels, preds)
        fmax = compute_fmax(time_labels.reshape(-1, len(terms)), preds)
        print(f'Test Loss - {time_loss}, AUC - {roc_auc}, Fmax - {fmax}')

    tdf['preds'] = list(preds)

    tdf.to_pickle('data/time_test_data_predictions.pkl')

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_fmax(labels, preds):
    fmax = 0.0
    patience = 0
    for t in range(0, 101):
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
    return fmax


def load_go_graph(terms_dict):
    go = Ontology('data/go.obo', with_rels=True)
    src = []
    dst = []
    for go_id, s in terms_dict.items():
        for g_id in go.get_parents(go_id):
            if g_id in terms_dict:
                src.append(s)
                dst.append(terms_dict[g_id])
    src = th.tensor(src)
    dst = th.tensor(dst)
    graph = dgl.graph((src, dst), num_nodes=len(terms_dict))
    graph = graph.add_self_loop()
    # dgl.save_graphs('data/go_mf.bin', graph)
    return graph

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
        
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DGGATModel(nn.Module):

    def __init__(self, go_graph, nb_iprs, nb_gos, device, hidden_dim=1024, embed_dim=1024, margin=0.1):
        super().__init__()
        self.go_graph = go_graph
        self.nb_gos = nb_gos
        input_length = nb_iprs
        net = []
        net.append(MLPBlock(input_length, hidden_dim))
        net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        self.net = nn.Sequential(*net)

        # Embeddings
        self.embed_dim = embed_dim
        self.go_embed = nn.Embedding(nb_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_bias = nn.Embedding(nb_gos, 1)
        nn.init.uniform_(self.go_bias.weight, -k, k)
        # self.go_embed.weight.requires_grad = False
        # self.go_rad.weight.requires_grad = False

        self.all_gos = th.arange(self.nb_gos).to(device)

        self.gat1 = GATConv(embed_dim, embed_dim, num_heads=1)
        
        
    def forward(self, features):
        x = self.net(features)
        go_embed = self.go_embed(self.all_gos)
        go_embed = self.gat1(self.go_graph, go_embed)
        go_embed = go_embed.view(self.nb_gos, self.embed_dim)
        go_bias = self.go_bias(self.all_gos).view(1, -1)
        x = th.matmul(x, go_embed.T) + go_bias
        logits = th.sigmoid(x)
        return logits

    
    
def load_data(data_file, terms_dict, iprs_dict, fold=1):
    gos = set(terms_dict)    
    df = pd.read_pickle(data_file)
    n = len(df)
    index = []
    for i, row in enumerate(df.itertuples()):
        ok = True
        annots = set(row.prop_annotations).intersection(gos)
        if len(annots) > 0:
            index.append(i)
    df = df.iloc[index]

    print('Num proteins', len(df))
    
    index = np.arange(len(df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(len(index) * 0.9)
    valid_n = int(train_n * 0.9)
    train_index = index[:valid_n]
    valid_index = index[valid_n:train_n]
    test_index = index[train_n:]

    train_df = df.iloc[train_index]
    train_df.to_pickle('data/train_data.pkl')
    valid_df = df.iloc[valid_index]
    valid_df.to_pickle('data/valid_data.pkl')
    test_df = df.iloc[test_index]
    test_df.to_pickle('data/test_data.pkl')

    train_data = get_data(train_df, iprs_dict, terms_dict)
    valid_data = get_data(valid_df, iprs_dict, terms_dict)
    test_data = get_data(test_df, iprs_dict, terms_dict)

    # Load time based split data
    tdf = pd.read_pickle('data/time_test_data.pkl')
    time_test_data = get_data(tdf, iprs_dict, terms_dict)
    
    return train_data, valid_data, test_data, test_df, time_test_data, tdf

def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
