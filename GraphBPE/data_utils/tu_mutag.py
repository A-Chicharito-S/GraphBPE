import torch
from torch_geometric.datasets.tu_dataset import *
from torch_geometric.data import Data
import requests
import os
from rdkit import Chem


def pre_transform_mutag(idx: int, data: Data, smiles: str):
    idx2node = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    x = data.x[:, -7:]  # 7 = |idx2node|, features of x = [node_features(if present) ||(concat) one-hot node label]
    # TODO: above behavior is defined by read_tu_data; however, with package upgrades, such a behavior might change
    assert torch.equal(x.sum(-1), torch.ones(size=(x.shape[0],)))
    # simple assertion on whether x is categorical/one-hot
    x_labels = torch.nonzero(x).tolist()  # (num_nodes, 2[row_idx, column_idx])
    node_names = [idx2node[idx[-1]] for idx in x_labels]
    data.node_names = node_names
    data.name = 'mutag_' + str(idx + 1)
    # assert smiles are the correct one
    adj, atom_types = smiles_to_adj_and_atom_types(smiles=smiles)
    assert len(atom_types) == len(node_names)
    assert ''.join(node_names) == ''.join(atom_types)
    assert torch.equal(edge_index_to_adjacency_matrix(edge_index=data.edge_index.T, num_atoms=len(atom_types)), adj)
    data.smiles = smiles
    return data


# smiles available at: https://ics.uci.edu/~baldig/learning/mutag/
def download_mutag_smiles(save_path):
    url = "https://ics.uci.edu/~baldig/learning/mutag/mutag_188_data.can"
    save_path = save_path + '/mutag_smiles.txt'
    if os.path.exists(path=save_path):
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("mutag smiles downloaded successfully.")
    else:
        print("Failed to download the mutag smiles. Status code:", response.status_code)


def get_smiles(save_path):
    save_path = save_path + '/mutag_smiles.txt'
    with open(save_path, 'r') as f:
        lines = f.readlines()
    smiles_list = [line.split()[0] for line in lines]
    return smiles_list


def smiles_to_adj_and_atom_types(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # return None, None
        # MUTAG 82 & 187 can not be parsed by RDKit, so we don not perform sanity checks on them
        # see the discussion here: https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CAEVXKcT_WUJQYhMzNvsT61DRPuFOEXaut46utVp3WsabQd89Cg%40mail.gmail.com/#msg36939142

    # Get atom types
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Get adjacency matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)

    return torch.tensor(adjacency_matrix), atom_types


def edge_index_to_adjacency_matrix(edge_index, num_atoms):
    adjacency_matrix = torch.zeros(num_atoms, num_atoms, dtype=torch.int)
    for edge in edge_index:
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Assuming undirected graph
    return adjacency_matrix


class AugmentedMutag(TUDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        assert name == 'MUTAG'
        super().__init__(root=root, name=name, transform=transform, pre_transform=pre_transform_mutag,
                         pre_filter=pre_filter, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr,
                         cleaned=cleaned)
        # call stack: AugmentedXXX ---> TUDataset init ---> InMemoryDataset init ---> Dataset init
        # ---> call self._process(), inside which call the override self.process() [defined below]
        # after the above is processed, begin to process node/edge features in the TUDataset init
        # (until now we can then have self.num_node_attributes)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            download_mutag_smiles(save_path=self.raw_dir)
            mutag_smiles = get_smiles(save_path=self.raw_dir)
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(idx=idx, data=d, smiles=mutag_smiles[idx]) for idx, d in
                             enumerate(data_list)]
                # add name to the Data object

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self._data.to_dict(), self.slices, sizes),
                   self.processed_paths[0])
