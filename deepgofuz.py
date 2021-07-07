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


@ck.command()
@ck.option(
    '--go-file', '-hf', default='data/go.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/train_data.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--rels-file', '-df', default='data/relations.pkl',
    help='List of relations and their ids')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--model-file', '-mf', default='data/deepgofuz.h5',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=12,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(go_file, data_file, rels_file, terms_file,  model_file, batch_size, epochs, load):
    global device
    device = 'cuda:1'
    gos_df = pd.read_pickle(terms_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}
    df = pd.read_pickle(data_file)

    rel_df = pd.read_pickle(rels_file)
    rels = {k: v for k, v in zip(rel_df['relations'].values, rel_df['ids'].values)}
    print(rels)
    subclass, hasfunc, relation  = load_data(data_file)
    dataset = MyDataset(df, subclass, hasfunc, relation)
    train_batches, train_steps = get_batches(dataset, batch_size)

    model = DGFuzModel(len(gos), len(rels))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        with ck.progressbar(train_batches) as bar:
            for data in bar:
                loss = model(data).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu()
                
            epoch_loss /= train_steps

        
        # model.eval()
        # valid_loss = 0
        # preds = []
        # with th.no_grad():
        #     with ck.progressbar(valid_nids) as bar:
        #         for n_id in bar:
        #             feat = annots[n_id, :].view(-1, 1)
        #             label = labels[n_id, :].view(1, -1)
        #             logits = model(g, etypes, feat)
        #             loss = loss_func(logits, label)
        #             valid_loss += loss.detach().cpu()
        #             preds = np.append(preds, logits.detach().cpu())
        #         valid_loss /= len(valid_nids)

        # valid_labels = labels[valid_nids, :].detach().cpu().numpy()
        # roc_auc = compute_roc(valid_labels, preds)
        # fmax = compute_fmax(valid_labels, preds.reshape(len(valid_nids), len(terms)))
        print(f'Epoch {epoch}: Loss - {epoch_loss}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_fmax(labels, preds):
    fmax = 0.0
    for t in range(1, 50):
        threshold = t / 100.0
        predictions = (preds >= threshold).astype(np.float32)
        tp = np.sum(labels * predictions, axis=1)
        fp = np.sum(predictions, axis=1) - tp
        fn = np.sum(labels, axis=1) - tp
        p = np.mean(tp / (tp + fp))
        r = np.mean(tp / (tp + fn))
        f = 2 * p * r / (p + r)
        if fmax < f:
            fmax = f
            patience = 0
        else:
            patience += 1
            if patience > 10:
                return fmax
    return fmax
        

class MyDataset(IterableDataset):

    def __init__(self, df, subclass, hasfunc, relation):
        self.df = df
        self.subclass = subclass
        self.sn = len(subclass)
        self.hasfunc = hasfunc
        self.hn = len(hasfunc)
        self.relation = relation
        self.rn = len(relation)
        self.n = max(self.sn, max(self.hn, self.rn))

    def get_data(self):
        si = 0
        hi = 0
        ri = 0
        for i in range(self.n):
            hi = i % self.hn
            si = i % self.sn
            ri = i % self.rn
            prot, hc = self.hasfunc[hi, 0], self.hasfunc[hi, 1]
            seq = self.df.iloc[int(prot)]['sequences']
            seq = to_onehot(seq)
            prot = th.from_numpy(seq).to(device)
            yield self.subclass[si], prot, hc, self.relation[ri]
            
    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.n

def get_batches(dataset, batch_size):
    steps = int(math.ceil(len(dataset) / batch_size))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return data_loader, steps

def collate(samples):
    subclass, prot, hc, relation = map(list, zip(*samples))
    hasfunc = th.stack(prot), th.stack(hc)
    return th.stack(subclass), hasfunc, th.stack(relation)


class DGFuzModel(nn.Module):

    def __init__(self, nb_gos, nb_rels, max_kernel=33, nb_filters=32, embedding_size=256):
        super().__init__()
        # DeepGOCNN
        kernels = range(8, max_kernel, 8)
        self.convs = []
        for kernel in kernels:
            self.convs.append((
                nn.Conv1d(21, nb_filters, kernel, device=device),
                nn.MaxPool1d(MAXLEN - kernel + 1)
            ))
        self.dg_out = nn.Linear(len(kernels) * nb_filters, embedding_size)

        # GO Class embeddings
        self.embed = nn.Embedding(nb_gos, embedding_size)
        self.rel_embed = nn.Embedding(nb_rels, embedding_size)
        self.top = th.zeros(embedding_size).to(device)
        self.top[0] = 1.0
        self.bot = th.zeros(embedding_size).to(device)
        self.hasfunc_id = th.tensor([8]).to(device)
        
    def deepgocnn(self, proteins):
        n = proteins.shape[0]
        output = []
        for conv, pool in self.convs:
            output.append(pool(conv(proteins)))
        output = th.cat(output, dim=1)
        return self.dg_out(output.view(n, -1))

    def loss(self, c, d):
        x = th.linalg.norm(c * (self.top - d), dim=1)
        return x

    def subclass_loss(self, data):
        c, d = data[:, 0], data[:, 1]
        c = self.embed(c)
        d = self.embed(d)
        return self.loss(c, d)

    def hasfunc_loss(self, data):
        p, c = data
        p = self.deepgocnn(p)
        r = self.rel_embed(self.hasfunc_id)
        c = self.embed(c)
        return self.loss(p, r + c)

    def relation_loss(self, data):
        c, r, d  = data[:, 0], data[:, 1], data[:, 2]
        c = self.embed(c)
        r = self.rel_embed(r)
        d = self.embed(d)
        return self.loss(c, r + d)
        
    def forward(self, data):
        subclass, hasfunc, relation = data
        loss = self.subclass_loss(subclass) + self.hasfunc_loss(hasfunc) + self.relation_loss(relation)
        return loss


def load_data(data_file):
    sub_df = pd.read_pickle('data/train_subclass.pkl')
    rel_df = pd.read_pickle('data/train_relations.pkl')
    hf_df = pd.read_pickle('data/train_hasfunc.pkl')

    subclass = np.array([*sub_df['subclass'].values])
    relation = np.array([*rel_df['relations'].values])
    hasfunc = np.array([*hf_df['hasfunc'].values])
    subclass = th.from_numpy(subclass).to(device)
    relation = th.from_numpy(relation).to(device)
    hasfunc = th.from_numpy(hasfunc).to(device)

    return subclass, hasfunc, relation

if __name__ == '__main__':
    main()