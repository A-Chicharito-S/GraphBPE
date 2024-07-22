import sys
import argparse
from collections import Counter
from .poly_hgraph import *
# from poly_hgraph import *
from rdkit import Chem
from multiprocessing import Pool
from tqdm import tqdm
import sys
import os
import sys

sys.path.append('..')
from data_utils import qm9, tu_mutag, tu_enzymes, tu_proteins, molecule_net


def process(data: dict):
    name2split = {}
    for name, s in tqdm(data.items()):
        hmol = MolGraph(s)
        node_split = [list(cls) for cls in hmol.clusters]
        if len(node_split) != 0:
            atom_ele = set([ele for split in node_split for ele in split])
            n_atoms = hmol.mol.GetNumAtoms()
            # in case of some weird situations, e.g., lipo_1647 O.O.O.CC(=O)O[C@@]12CO[C@@H]1C[C@H](O)[C@]3(C)[C@@H]2[C@H](OC(=O)c4ccccc4)[C@]5(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c6ccccc6)C(=C([C@@H](O)C3=O)C5(C)C)C
            if len(atom_ele) < n_atoms:  # some nodes are not included
                total_atom = [i for i in range(n_atoms)]
                left_out = set(total_atom) - atom_ele  # in the total atoms but not present in node split
                for left_out_atom in left_out:
                    node_split.append([left_out_atom])

                atom_ele = set([ele for split in node_split for ele in split])  # check the new atom_ele set
            assert max(atom_ele) == n_atoms - 1
            assert len(atom_ele) == n_atoms
        name2split[name] = node_split
    return name2split


def fragment_process(data):
    counter = Counter()
    for smiles in tqdm(data):
        mol = get_mol(smiles)
        if mol is None:
            continue
        fragments = find_fragments(mol)
        for fsmiles, _ in fragments:
            counter[fsmiles] += 1
    return counter


def tokenize_dataset(name2smi_dict: dict, ncpu=16, min_frequency=100):
    smiles_data = [val for val in name2smi_dict.values()]
    smiles_data = list(set(smiles_data))
    print('num of unique smiles: {}'.format(len(smiles_data)))

    batch_size = len(smiles_data) // ncpu + 1
    batches = [smiles_data[i: i + batch_size] for i in range(0, len(smiles_data), batch_size)]

    # get the frequent fragments (freq>=100)
    pool = Pool(ncpu)
    counter_list = pool.map(fragment_process, batches)
    counter = Counter()
    for cc in counter_list:
        counter += cc

    fragments = [fragment for fragment, cnt in counter.most_common() if cnt >= min_frequency]
    MolGraph.load_fragments(fragments)

    def split_dict(dictionary, n):
        # Get the list of dictionary items (key-value pairs)
        items = list(dictionary.items())
        batch_size = len(items) // n + 1
        chunks = [dict(items[i: i + batch_size]) for i in range(0, len(items), batch_size)]
        return chunks

    dict_batches = split_dict(dictionary=name2smi_dict, n=ncpu)
    pool = Pool(ncpu)
    name2split_list = pool.map(process, dict_batches)
    name2split_dict = {}
    for batch_name2split in name2split_list:
        name2split_dict.update(batch_name2split)
    assert len(name2split_dict) == len(name2smi_dict)
    return name2split_dict


# if __name__ == "__main__":
#
#     dataset = tu_mutag.AugmentedMutag(root='dataset', name='MUTAG')
#     # name2smi_dict = {i: 'CCCC' for i in range(1000)}
#     name2smi_dict = {}
#     for data in dataset:
#         name, smiles = data.name, data.smiles
#         name2smi_dict[name] = smiles
#     results = tokenize_dataset(name2smi_dict=name2smi_dict, ncpu=2, min_frequency=100)
#
#     print(results['mutag_1'])
