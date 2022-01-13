from matplotlib import pyplot as plt
import click as ck
import numpy as np

@ck.command()
def main():
    sims = list()
    cnt = 0
    with open('data/swissprot_exp.sim') as f:
        for line in f:
            it = line.strip().split('\t')
            p1, p2, sim = it[0], it[1], float(it[2])
            if p1 == p2:
                continue
            sims.append(sim)
            if sim >= 80:
                cnt += 1
    print(cnt)
    plt.hist(sims, color='blue', bins=10)
    plt.title('Distribution of sequence similarity values')
    plt.xlabel('Identity (%)')
    plt.ylabel('Number of similar protein pairs')
    plt.savefig('sim-hist.eps')
    

if __name__ == '__main__':
    main()
