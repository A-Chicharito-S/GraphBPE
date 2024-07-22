import sys
import argparse 
from collections import Counter
from poly_hgraph import *
from rdkit import Chem
from multiprocessing import Pool
from tqdm import tqdm

def process(data):
    vocab = set()
    for line in tqdm(data):
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

def fragment_process(data):
    counter = Counter()
    for smiles in tqdm(data):
        mol = get_mol(smiles)
        fragments = find_fragments(mol)
        for fsmiles, _ in fragments:
            counter[fsmiles] += 1
    return counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_frequency', type=int, default=100)
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--data', type=str, default='/zfsauton2/home/yuchens3/hgraph2graph-master/data/polymers/all.txt')
    parser.add_argument('--output', type=str, default='vocab.txt ')
    args = parser.parse_args()

    import os

    cwd = os.getcwd()

    data = []
    with open(args.data, 'r') as f:
        for line in f.readlines():
            for mol in line.split()[:2]:
                data.append(mol)
    data = list(set(data))[:2000]

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    counter_list = pool.map(fragment_process, batches)
    counter = Counter()
    for cc in counter_list:
        counter += cc
    
    fragments = [fragment for fragment,cnt in counter.most_common() if cnt >= args.min_frequency]
    MolGraph.load_fragments(fragments)

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    fragments = set(fragments)
    f = open(args.output, 'w')
    for x,y in sorted(vocab):
        cx = Chem.MolToSmiles(Chem.MolFromSmiles(x))  # dekekulize
        # print(x, cx)
        # print(x, y, cx in fragments)
        f.write('{} {} {}'.format(x, y, cx in fragments)+'\n')

