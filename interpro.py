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
from torch.utils.data import DataLoader, IterableDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from dgl.nn import GraphConv
import dgl

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/interpro_train_data.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/interpros.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--model-file', '-mf', default='data/interpro.th',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=64,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(data_file, terms_file,  model_file, batch_size, epochs, load):
    global device
    device = 'cuda:1'
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['interpros'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    df = pd.read_pickle(data_file)

    net = IPRModel(len(terms))
    net = net.to(device)
    loss_func = nn.BCELoss()
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    
    print('Loading data')
    train_df, valid_df = load_data(data_file)
    train_dataset = MyDataset(train_df, terms_dict)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
    )
    valid_dataset = MyDataset(valid_df, terms_dict)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=False,
    )
    print(len(train_dataset), len(valid_dataset))
    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_df) / batch_size))
            with ck.progressbar(length=train_steps) as bar:
                for train_features, train_labels in dataloader:
                    bar.update(1)
                    logits = net(train_features.to(device))
                    loss = F.binary_cross_entropy(logits, train_labels.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    
            train_loss /= train_steps

            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_df) / batch_size))
                valid_loss = 0
                preds = []
                labels = []
                with ck.progressbar(length=valid_steps) as bar:
                    for valid_features, valid_labels in valid_dataloader:
                        bar.update(1)
                        logits = net(valid_features.to(device))
                        batch_loss = F.binary_cross_entropy(logits, valid_labels.to(device))
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                        labels = np.append(labels, valid_labels.detach().cpu().numpy())
                valid_loss /= valid_steps
                preds = preds.reshape(len(valid_df), len(terms))
                labels = labels.reshape(len(valid_df), len(terms))
                roc_auc = compute_roc(labels, preds)
                fmax = compute_fmax(labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}, Fmax - {fmax}')

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model')
                    th.save(net.state_dict(), model_file)

    # Loading best model
    # print('Loading the best model')
    # net.load_state_dict(th.load(model_file))
    # net.eval()
    # with th.no_grad():
    #     test_steps = int(math.ceil(len(test_nids) / batch_size))
    #     test_loss = 0
    #     preds = []
    #     for input_nodes, output_nodes, blocks in test_dataloader:
    #         logits = net(blocks, features[input_nodes])
    #         batch_loss = F.binary_cross_entropy(logits, labels[output_nodes])
    #         test_loss += batch_loss.detach().item()
    #         preds = np.append(preds, logits.detach().cpu().numpy())
    #     test_loss /= test_steps
    #     preds = preds.reshape(len(test_nids), len(terms))
    #     test_labels = labels[test_nids].detach().cpu().numpy()
    #     roc_auc = compute_roc(test_labels, preds)
    #     fmax = compute_fmax(test_labels, preds)
    #     print(f'Test Loss - {test_loss}, AUC - {roc_auc}, Fmax - {fmax}')

    # test_df['preds'] = list(preds)

    # test_df.to_pickle(out_file)

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_fmax(labels, preds):
    fmax = 0.0
    patience = 0
    for t in range(1, 101):
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
        

class IPRModel(nn.Module):

    def __init__(self, nb_iprs, max_kernel=129, nb_filters=16, hidden_dim=1024):
        super().__init__()
        # DeepGOCNN
        kernels = range(8, max_kernel, 8)
        self.convs = []
        for kernel in kernels:
            self.convs.append((
                nn.Conv1d(21, nb_filters, kernel, device=device),
                nn.MaxPool1d(MAXLEN - kernel + 1)
            ))
        self.dg_out = nn.Linear(len(kernels) * nb_filters, hidden_dim)
        
        # GO Class embeddings
        self.fc = nn.Linear(len(kernels) * nb_filters, nb_iprs)
        self.dropout = nn.Dropout()
        
    def deepgocnn(self, proteins):
        n = proteins.shape[0]
        output = []
        for conv, pool in self.convs:
            x = pool(conv(proteins))
            output.append(x)
        output = th.cat(output).view(n, -1)
        return output
        #return self.dropout(th.relu(self.dg_out(output.view(n, -1))))
        
    def forward(self, x):
        x = self.deepgocnn(x)
        x = th.sigmoid(self.fc(x))
        return x



class MyDataset(IterableDataset):

    def __init__(self, df, terms_dict):
        self.df = df
        self.terms_dict = terms_dict
        self.n = len(df)

    def get_data(self):
        for i, row in enumerate(self.df.itertuples()):
            seq = row.sequences
            seq = to_onehot(seq)
            prot = th.from_numpy(seq)
            label = np.zeros(len(self.terms_dict), dtype=np.float32)
            for ipr_id in row.interproscan:
                if ipr_id in self.terms_dict:
                    label[self.terms_dict[ipr_id]] = 1
            label = th.from_numpy(label)
            yield prot, label
        
    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.n

    
def load_data(data_file, fold=1):
    df = pd.read_pickle(data_file)
    n = len(df)
    
    index = np.arange(len(df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(n * 0.95)
    train_index = index[:train_n]
    valid_index = index[train_n:]

    train_df = df.iloc[train_index]
    valid_df = df.iloc[valid_index]
    
    return train_df, valid_df

if __name__ == '__main__':
    main()
