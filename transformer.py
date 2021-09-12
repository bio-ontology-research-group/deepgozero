import click as ck
import pandas as pd
from utils import Ontology
import torch
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset
from itertools import cycle
import math
from aminoacids import to_onehot, to_tokens, MAXLEN
from dgl.nn import GraphConv
import dgl
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    '--model-file', '-mf', default='data/deepgogcn_mf.th',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=12,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--out-file', '-of', default='data/predictions_deepgogcn.pkl',
    help='Prediction model')
def main(go_file, data_file, terms_file,  model_file, batch_size, epochs, load, out_file):
    global device
    device = 'cuda:1'
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Loading data')
    ppi_g, features, labels, train_nids, valid_nids, test_nids, test_df = load_data(data_file, terms_dict)
    
    net = DGTransformerModel(21, len(terms), 2, 2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.BCELoss()

    labels = labels.to(device)
    # features = features.to(device)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.NodeDataLoader(
        ppi_g, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        device=device
    )

    valid_dataloader = dgl.dataloading.NodeDataLoader(
        ppi_g, valid_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        device=device
    )
    test_dataloader = dgl.dataloading.NodeDataLoader(
        ppi_g, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        device=device
    )
    
    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_nids) / batch_size))
            with ck.progressbar(length=train_steps) as bar:
                for input_nodes, output_nodes, blocks in dataloader:
                    bar.update(1)
                    logits = net(features[:, output_nodes].to(device))
                    loss = F.binary_cross_entropy(logits, labels[output_nodes])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    
            train_loss /= train_steps

            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_nids) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps) as bar:
                    for input_nodes, output_nodes, blocks in valid_dataloader:
                        bar.update(1)
                        logits = net(features[:, output_nodes].to(device))
                        batch_loss = F.binary_cross_entropy(logits, labels[output_nodes])
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                valid_labels = labels[valid_nids].detach().cpu().numpy()
                roc_auc = compute_roc(valid_labels, preds)
                fmax = compute_fmax(valid_labels, preds.reshape(len(valid_nids), len(terms)))
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
        test_steps = int(math.ceil(len(test_nids) / batch_size))
        test_loss = 0
        preds = []
        for input_nodes, output_nodes, blocks in test_dataloader:
            logits = net(eatures[:, output_nodes].to(device))
            batch_loss = F.binary_cross_entropy(logits, labels[output_nodes])
            test_loss += batch_loss.detach().item()
            preds = np.append(preds, logits.detach().cpu().numpy())
        test_loss /= test_steps
        preds = preds.reshape(len(test_nids), len(terms))
        test_labels = labels[test_nids].detach().cpu().numpy()
        roc_auc = compute_roc(test_labels, preds)
        fmax = compute_fmax(test_labels, preds)
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
        

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class DGTransformerModel(nn.Module):

    def __init__(self, ntoken: int, ngos: int, nhead: int, nlayers: int,
                 d_model: int = 64, d_hid: int = 64, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.pool = nn.MaxPool1d(8)
        self.decoder = nn.Linear(d_model * 250, d_hid)
        self.out = nn.Linear(d_hid, ngos)
        self.init_weights()
        
    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.transpose(output, 0, 1)
        output = torch.transpose(output, 1, 2)
        bs = output.shape[0]
        output = self.pool(output).reshape(bs, -1)
        output = th.relu(self.decoder(output))
        output = th.sigmoid(self.out(output))
        return output
    

def load_data(data_file, terms_dict, fold=1):
    from dgl import save_graphs, load_graphs
    df = pd.read_pickle(data_file)
    n = len(df)
    graphs, data_dict = dgl.load_graphs('data/ppi.bin')
    ppi_g = graphs[0]
    ipr_df = pd.read_pickle('data/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}
    
    proteins = df['proteins']
    prot_idx = {v: k for k, v in enumerate(proteins)}
    labels = np.zeros((n, len(terms_dict)), dtype=np.float32)
    features = np.zeros((MAXLEN, n), dtype=np.int32)
    # Filter proteins with annotations
    for i, row in enumerate(df.itertuples()):
        for go_id in row.dg_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1
        seq = row.sequences
        seq = to_tokens(seq)
        # seq = to_onehot(seq)
        features[:, i] = seq
        # for ipr in row.interpros:
        #     features[i, iprs_dict[ipr]] = 1

    features = th.LongTensor(features)
    labels = th.FloatTensor(labels)
    

    index = np.arange(len(df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(len(index) * 0.95)
    valid_n = int(train_n * 0.95)
    train_index = index[:valid_n]
    valid_index = index[valid_n:train_n]
    test_index = index[train_n:]

    train_df = df.iloc[np.concatenate([train_index, valid_index])]
    train_df.to_pickle('data/train_data.pkl')
    test_df = df.iloc[test_index]
    
    train_index = torch.LongTensor(train_index)
    valid_index = torch.LongTensor(valid_index)
    test_index = torch.LongTensor(test_index)
    
    return ppi_g, features, labels, train_index, valid_index, test_index, test_df

if __name__ == '__main__':
    main()
