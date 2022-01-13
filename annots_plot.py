from matplotlib import pyplot as plt
import click as ck
import numpy as np
from collections import Counter
import pandas as pd

@ck.command()
def main():
    counts = Counter()
    annots = {}
    cnt = 0
    with open('zero_completely.txt') as f:
        for line in f:
            it = line.strip().split()
            if len(it) > 2:
                go_id, n = it[0], int(it[2])
                annots[go_id] = n
                counts[n] += 1
    counts = [v for c, v in counts.most_common(10)]
    counts = pd.Series(counts)
    perfs = get_average_performance(annots)
    print(perfs)
    ax = counts.plot(kind="bar")
    rects = ax.patches
    labels = ['',] + perfs
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        )
    ax.set_xticklabels(np.arange(10), rotation=0)
    ax.set_title('Distribution of classes with annotations < 10')
    ax.set_xlabel('Number of annotations')
    ax.set_ylabel('Number of classes')
    
    plt.savefig('annots-num.eps')


def get_average_performance(annots):
    aucs = {}
    with open('data/results/result_zero_10.txt') as f:
        for line in f:
            if not line.startswith('GO:'):
                continue
            it = line.strip().split()
            go_id, auc = it[0], float(it[1])
            if go_id in annots:
                n = annots[go_id]
                if n not in aucs:
                    aucs[n] = []
                aucs[n].append(auc)
    avgs = []
    for i in range(1, 10):
        avgs.append(f'{np.mean(aucs[i]):.3f}')
    return avgs

if __name__ == '__main__':
    main()
