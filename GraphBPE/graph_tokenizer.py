import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import os
import torch
from tqdm import tqdm
import json
from copy import deepcopy
import hydra
from omegaconf import DictConfig
import pickle
from supported import supported_dataset


# 1. shrink all the rings ---> keep each mol as: {'adj': xxx, 'node_name': xxx}
# 2. for each mol, find co-occurring pairs
#    ---> keep the (start, end) and its symbol in the node (& globally[e.g., "C=O" should be treated the same as "O=C"])
# 3. count the co-occurrence of the patterns {and where to find them [e.g., mol_id]}
# 4. find the most appeared pair
# 5. contract them

class GraphBPE:
    def __init__(self, dataset_cfg):
        # if contract_rings: contract rings in the preprocessing step
        self.contract_rings = dataset_cfg.contract_rings
        self.contract_cliques = dataset_cfg.contract_cliques
        self.min_nodes_in_cliques = dataset_cfg.min_nodes_in_cliques
        self.consider_outside_neighbors = dataset_cfg.consider_outside_neighbors
        self.num_round = dataset_cfg.num_round
        self.dataset_name = dataset_cfg.name

        if self.dataset_name in supported_dataset.keys():
            self.dataset = supported_dataset[self.dataset_name]
        else:
            raise NotImplementedError('{} is not implemented'.format(self.dataset_name))

        self.prev_tok_dataset = None
        self.tokenized_dataset = {}
        # {'mol_name': {'adj': xxx, 'node_names': [...], 'node_features': xxx, 'node_split': [[..], [], ...]}, ...}
        self.prev_vocab_freq = None
        self.com2sim_node_names = {}  # {'round_n': {'(complex) i-th hyper-node name': 'nRi', ...}, ...}
        self.vocab_freq = {}  # {'node_name': frequency} keep track of the sub-graph patterns and frequencies
        self.co_occurrence = {}
        # {'co-occurred-pair name': {'freq': xx,
        #                            'mols': {'molecule name': [(s_node, e_node), ...], ...}
        #                            },
        #                            ...}

        self.vocab_size_tok = []  # keep track of the vocab size as tokenization rounds increases
        self.vocab_size_origin = len(self.dataset[0].x[0])  # (one-hot) feature size of the 1st node of the 1st molecule
        self.vocab_origin = set()
        self.graph_grammar = {i: [] for i in range(self.num_round + 1)}
        if self.contract_cliques:
            self.graph_grammar['c'] = []

        self.num_of_hyper_node = 0
        # keep track of the number of hyper-nodes so far; thus, we can convert the i-th (sophisticated) hyper-node name
        # into: Round_n_i
        self.preprocessing()

    def save_tokenized_dataset(self, round_n):
        save_dir = 'tokenized_dataset/' + '{}_contract_rings_{}_cliques_{}'.format(self.dataset_name,
                                                                                   str(self.contract_rings),
                                                                                   str(self.contract_cliques))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + '/{}.pickle'.format(round_n), 'wb') as f:
            if round_n == 0:
                if self.contract_rings and self.contract_cliques:
                    com2sim = {'clique': self.com2sim_node_names['Cliques'], 'ring': self.com2sim_node_names['Rings']}
                elif self.contract_rings and not self.contract_cliques:
                    com2sim = self.com2sim_node_names['Rings']
                elif self.contract_cliques and not self.contract_rings:
                    com2sim = self.com2sim_node_names['Cliques']
                else:
                    com2sim = None

            else:
                com2sim = self.com2sim_node_names['Round_{}'.format(round_n)]
            data_infor = {'dataset': self.tokenized_dataset, 'grammar': self.graph_grammar,
                          'vocab': {'ori_size': self.vocab_size_origin, 'ori_tok': self.vocab_origin,
                                    'tok_sizes': self.vocab_size_tok},
                          'complex2simple': com2sim}
            pickle.dump(data_infor, f, protocol=pickle.HIGHEST_PROTOCOL)

    def preprocessing(self):
        print('=============begin preprocessing=============')
        if self.contract_rings:
            # init the simple-to-complex hyper-node name mapping
            self.com2sim_node_names['Rings'] = {}
        if self.contract_cliques:
            self.com2sim_node_names['Cliques'] = {}

        for data in tqdm(self.dataset):
            name, node_names, node_features = data.name, data.node_names, data.x
            node_features = node_features.numpy()
            adj = nx.adjacency_matrix(to_networkx(data, to_undirected=True)).todense()
            # see also: https://github.com/pyg-team/pytorch_geometric/issues/2827
            node_split = [[i] for i in range(len(adj))]

            self.tokenized_dataset[name] = {'adj': adj, 'node_names': node_names,
                                            'node_features': node_features,
                                            'node_split': node_split}

        if self.contract_cliques:
            for name in tqdm(self.tokenized_dataset.keys()):
                inst = self.tokenized_dataset[name]
                adj, node_names, node_features, node_split = inst['adj'], inst['node_names'], inst['node_features'], \
                                                             inst['node_split']

                cliques_to_contract = self.get_cliques_and_adj(adj=adj, min_num_nodes=self.min_nodes_in_cliques)

                adj, node_names, node_features, node_split = self.contract_clique(adjacency_matrix=adj,
                                                                                  contracted_nodes=cliques_to_contract,
                                                                                  node_names=node_names,
                                                                                  node_features=node_features,
                                                                                  node_split=node_split,
                                                                                  consider_outside_neighbors=False)
                # set consider_outside_neighbors=True to better reconstruct the rings
                # self.vocab_freq is updated here
                self.tokenized_dataset[name] = {'adj': adj, 'node_names': node_names, 'node_features': node_features,
                                                'node_split': node_split}

                # self.get_pairs_and_count(adjacency_matrix=adj, node_names=node_names, mol_name=name)
                # update co-occurrence

            # reset the number of hyper-nodes to 0 (at round n)
            self.num_of_hyper_node = 0

        self.assert_node_split()

        for i, name in enumerate(tqdm(self.tokenized_dataset.keys())):
            inst = self.tokenized_dataset[name]
            adj, node_names, node_features, node_split = inst['adj'], inst['node_names'], inst['node_features'], inst[
                'node_split']

            if self.contract_rings:  # (continue to) contract rings
                rings_to_contract = self.get_rings_and_adj(adj=adj)

                adj, node_names, node_features, node_split = self.contract_ring(adjacency_matrix=adj,
                                                                                contracted_nodes=rings_to_contract,
                                                                                node_names=node_names,
                                                                                node_features=node_features,
                                                                                node_split=node_split,
                                                                                consider_outside_neighbors=self.consider_outside_neighbors)
                # set consider_outside_neighbors=True to better reconstruct the rings
                # self.vocab_freq is updated here
                self.tokenized_dataset[name] = {'adj': adj, 'node_names': node_names, 'node_features': node_features,
                                                'node_split': node_split}

                self.get_pairs_and_count(adjacency_matrix=adj, node_names=node_names, mol_name=name)
                # update co-occurrence
            else:
                self.get_pairs_and_count(adjacency_matrix=adj, node_names=node_names, mol_name=name)
                # update co-occurrence (two situations: 1. only clique is contracted; thus, need update /
                # 2. not clique or ring is contracted; thus, need update+vocab_freq update)

                # update normal-node vocab (there are no hyper-node yet)
                if not self.contract_cliques:  # [in this branch, self.contract_rings is False]
                    for node_name in node_names:
                        if node_name not in self.vocab_freq.keys():
                            self.vocab_freq[node_name] = 1
                        else:
                            self.vocab_freq[node_name] += 1

        self.auto_count()

        self.assert_node_split()

        # reset the number of hyper-nodes to 0 (at round n)
        self.num_of_hyper_node = 0

        self.vocab_freq = {k: v for (k, v) in self.vocab_freq.items() if v > 0}
        print('current vocab-freq:\n{} {}'.format(len(self.vocab_freq), self.vocab_freq))
        print('=============preprocessing ends=============')

        self.vocab_size_tok.append(len(self.vocab_freq))
        self.save_tokenized_dataset(round_n=0)

    def assert_vocab(self):  # assert whether the tokenization is correct at vocabulary level
        prev = set(self.prev_vocab_freq.keys())
        curr = set(self.vocab_freq.keys())

        if len(prev) == len(curr) and len(prev & curr) == len(prev):
            print('prev vocab = current vocab; error')
            exit()

    def assert_tokenized_dataset(self):  # assert whether the tokenization is correct at dataset level
        if len(self.tokenized_dataset) != len(self.prev_tok_dataset):
            return
        cnt = 0
        for mol_name in self.tokenized_dataset.keys():
            curr_adj = self.tokenized_dataset[mol_name]['adj']
            prev_adj = self.prev_tok_dataset[mol_name]['adj']
            if np.array_equal(curr_adj, prev_adj):
                cnt += 1
        if cnt == len(self.tokenized_dataset):
            print('prev tok dataset adj = current tok dataset adj; error')
            exit()

    def auto_count(self):
        # check the differences between self.vocab_freq and the true vocab (derived from self.tokenized_dataset)
        vocab_freq = {}
        for mol_name in self.tokenized_dataset.keys():
            for node in self.tokenized_dataset[mol_name]['node_names']:
                if node not in vocab_freq.keys():
                    vocab_freq[node] = 1
                else:
                    vocab_freq[node] += 1
        for mol_name in vocab_freq.keys():
            if mol_name not in self.vocab_freq.keys():
                print('{} not in true vocab'.format(mol_name))
            if vocab_freq[mol_name] != self.vocab_freq[mol_name]:
                print('{}, true freq {}; counted freq {}; gap: {}'.format(mol_name,
                                                                          vocab_freq[mol_name],
                                                                          self.vocab_freq[mol_name],
                                                                          vocab_freq[mol_name] - self.vocab_freq[
                                                                              mol_name]))

    def assert_node_split(self):  # assert whether the tokenization is correct at node level
        for data in self.dataset:
            x = data.x
            mol_name = data.name
            tokenized_mol = self.tokenized_dataset[mol_name]
            for idx in range(len(tokenized_mol['node_features'])):
                rdm_node_idx = idx  # random.randint(0, len(tokenized_mol['node_features']) - 1)
                sum_one_hot = tokenized_mol['node_features'][rdm_node_idx]
                node_split = tokenized_mol['node_split'][rdm_node_idx]
                test_sum_one_hot = 0
                for node in node_split:
                    test_sum_one_hot += x[node]
                assert np.array_equal(sum_one_hot, test_sum_one_hot.numpy())

    def BPE_on_graph(self, round_n):
        most_freq = sorted(self.co_occurrence.items(), key=lambda item: item[1]['freq'], reverse=True)[0]
        pair_name, freq, mols = most_freq[0], most_freq[1]['freq'], most_freq[1]['mols']
        # mols: {'molecule name': [(s_node, e_node), ...], ...}

        # init the simple-to-complex hyper-node name mapping
        self.com2sim_node_names['Round_{}'.format(round_n)] = {}
        for mol_name in mols.keys():
            adj, node_names, node_features, node_split = self.tokenized_dataset[mol_name]['adj'], \
                                                         self.tokenized_dataset[mol_name]['node_names'], \
                                                         self.tokenized_dataset[mol_name]['node_features'], \
                                                         self.tokenized_dataset[mol_name]['node_split']

            edges_to_contract = self.find_edges_to_merge(nodes_list=mols[mol_name])
            assert len(edges_to_contract) != 0
            adj, node_names, node_features, node_split = self.contract_edge(adjacency_matrix=adj,
                                                                            contracted_nodes=edges_to_contract,
                                                                            node_names=node_names,
                                                                            node_features=node_features,
                                                                            node_split=node_split,
                                                                            round_n=round_n,
                                                                            consider_outside_neighbors=self.consider_outside_neighbors)
            # self.vocab_freq is also updated
            self.tokenized_dataset[mol_name] = {'adj': adj, 'node_names': node_names, 'node_features': node_features,
                                                'node_split': node_split}
            # update the tokenization information

        # reset the co_occurrence matrix to empty, update it after merging
        self.co_occurrence = {}
        # reset the number of hyper-nodes to 0 (at round n)
        self.num_of_hyper_node = 0

        for mol_name in self.tokenized_dataset.keys():
            adj, node_names, node_features = self.tokenized_dataset[mol_name]['adj'], \
                                             self.tokenized_dataset[mol_name]['node_names'], \
                                             self.tokenized_dataset[mol_name]['node_features']
            self.get_pairs_and_count(adjacency_matrix=adj, node_names=node_names, mol_name=mol_name)
            # update co-occurrences

        self.auto_count()

    def process(self):
        round_n = 0
        for _ in tqdm(range(self.num_round)):
            # print('round: {}'.format(round_n + 1))

            self.prev_tok_dataset = deepcopy(self.tokenized_dataset)
            self.prev_vocab_freq = deepcopy(self.vocab_freq)
            round_n += 1
            self.BPE_on_graph(round_n=round_n)  # update vocab_freq etc.

            self.assert_node_split()
            self.auto_count()

            self.vocab_freq = {k: v for (k, v) in self.vocab_freq.items() if v > 0}

            self.assert_vocab()
            self.assert_tokenized_dataset()

            self.vocab_size_tok.append(len(self.vocab_freq))
            self.save_tokenized_dataset(round_n=round_n)

            if len(self.co_occurrence) == 0:
                break

    @staticmethod
    def find_edges_to_merge(nodes_list: list[list[int, int]]):
        existing_nodes = []
        contract_edges = []
        for edge in nodes_list:
            start_node, end_node = edge
            if start_node not in existing_nodes and end_node not in existing_nodes:
                existing_nodes += edge
                contract_edges.append(edge)
        return contract_edges

    def get_rings_and_adj(self, adj: np.ndarray):
        # given a molecule, contract all the rings
        G = nx.from_numpy_array(A=adj).to_undirected()
        # see: https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.to_networkx
        chordless_cycles = list(nx.chordless_cycles(G, length_bound=10))  # List[List[Int]]

        # a counter example: (mutag data 22):
        # extracted chordless cycles with length_bound=None
        # [[1, 0, 5, 4, 3, 2], [4, 3, 9, 8, 7, 6], [9, 8, 17, 16, 11, 10],
        # [9, 8, 17, 18, 19, 15, 14, 13, 12, 11, 10], [12, 11, 16, 15, 14, 13], [16, 15, 19, 18, 17]]
        # where a very long loop actually contains 3 rings, to avoid this case, we simply bound its length (<=10)
        # TODO: note that this solution (specifying length_bound) is not optimal and could cause potential problems

        num_cycles = len(chordless_cycles)
        # shared_nodes_adj_matrix = np.zeros(shape=(num_cycles, num_cycles))

        rings_to_remove = []
        stored_pairs_of_the_triplet = []  # say (i, j, k) [r1, r2, outer-ring] is a triplet discovered in checking
        # (i, j), we store (i, j), (i, k), (j, k), such that when we meet (i, k) or (j, k), we don't need to check again

        for i in range(num_cycles):
            cycle_i = set(chordless_cycles[i])
            for j in range(i + 1, num_cycles):
                if {i, j} in stored_pairs_of_the_triplet:
                    continue
                cycle_j = set(chordless_cycles[j])
                shared_nodes = cycle_i & cycle_j
                if len(shared_nodes) <= 2:  # only sharing 1 node / 1 edge (2 nodes), by doing this, we can't deal with
                    # some special cases, such as there might be situations where a Mercedes-Benz alike structure,
                    # and the outer loop should be removed
                    continue
                all_nodes = cycle_i | cycle_j
                third_cycle_idx = None
                third_cycle = None
                for k in range(num_cycles):  # as long as there exists "len(shared_nodes) > 2" edges, we can
                    # always find a triplet (r1, r2, outer-ring)
                    if k == i or k == j:
                        continue
                    cycle_k = set(chordless_cycles[k])
                    if cycle_k.issubset(all_nodes):
                        third_cycle_idx = k
                        third_cycle = cycle_k
                        break
                assert third_cycle_idx is not None
                stored_pairs_of_the_triplet.extend([{i, j}, {i, third_cycle_idx}, {j, third_cycle_idx}])

                def get_sum_of_degree(set_of_nodes, adjacency_matrix):
                    sum_of_degree = 0
                    for shared_n in set_of_nodes:
                        degree = adjacency_matrix[shared_n].sum()
                        sum_of_degree += degree
                    return sum_of_degree

                edge_degree_ij = get_sum_of_degree(set_of_nodes=shared_nodes, adjacency_matrix=adj)
                edge_degree_ik = get_sum_of_degree(set_of_nodes=cycle_i & third_cycle, adjacency_matrix=adj)
                edge_degree_jk = get_sum_of_degree(set_of_nodes=cycle_j & third_cycle, adjacency_matrix=adj)
                remove_idx = sorted([[edge_degree_ij, third_cycle_idx], [edge_degree_ik, j], [edge_degree_jk, i]],
                                    key=lambda x: x[0])[0][-1]  # sort by minimum degree, take corresponding idx
                # the nodes with the minimum degrees (e.g., i, j) are those we want to keep (meaning they reside on the
                # two sides of the "len(shared_nodes) > 2" edges), which means we will remove the rest cycle (e.g., k)
                # in the triplet. It might be somewhat problematic but somehow works, such an example: enzymes_2,
                # after contract_clique: cycle_1: {0, 1, 18, 19} cycle_2: {0, 2, 3, 4, 18, 19, 20}
                # third cycle: {1, 2, 3, 4}; cycle_2 is selected to remove [this is also the reason we in line 507 we
                # use for k in range(num_cycles) instead of "in rang(j+1, num_cycles)"]

                rings_to_remove.append(remove_idx)

        chordless_cycles = np.delete(chordless_cycles, rings_to_remove, axis=0).tolist()

        return chordless_cycles

    @staticmethod
    def get_cliques_and_adj(adj, min_num_nodes=None):
        if min_num_nodes is None:
            min_num_nodes = 3
        G = nx.from_numpy_array(A=adj).to_undirected()
        cliques = []
        for c in nx.find_cliques_recursive(G):
            if len(c) < min_num_nodes:
                continue
            cliques.append(c)
        return cliques

    def contract_edge(self, adjacency_matrix: np.ndarray, contracted_nodes: list[list[int, ...]], node_names: list,
                      node_features: np.ndarray, node_split: list[list[int]], round_n: int,
                      consider_outside_neighbors=True):
        adj_len, node_names_len = len(adjacency_matrix), len(node_names)
        assert adj_len == node_names_len, \
            'length of adjacency matrix ({}) does not match node names ({})'.format(adj_len, node_names_len)
        num_hyper_nodes = len(contracted_nodes)

        if num_hyper_nodes == 0:
            return adjacency_matrix, node_names, node_features, node_split

        deleting_nodes = set()  # keep track of the nodes that constitute hyper-nodes

        # find the neighbors of each hyper_node
        hyper2normal = []  # stores adjacencies between hyper-nodes and normal-nodes
        hyper2hyper = np.zeros(shape=(num_hyper_nodes, num_hyper_nodes))
        # stores adjacencies between hyper-nodes and hyper-nodes
        hyper_names = []
        hyper_features = []  # (num_hyper_nodes, feature_dim)
        hyper_and_neighbors = []

        for i in range(num_hyper_nodes):

            hyper_node = contracted_nodes[i]
            # during init, this can be a list consisting of multiple node indices (ring)
            # in other situations, it only contains two nodes

            neighbors, hyper_node_name, hyper_feature = self.get_hyper_node_neighbors_name_feature(
                adjacency_matrix=adjacency_matrix,
                hyper_node=hyper_node,
                node_names=node_names, node_features=node_features,
                consider_outside_neighbors=consider_outside_neighbors)
            # neighbors: 0-1 vectors indicating both the hyper nodes and the neighboring nodes in the adjacency matrix
            hyper2normal.append(neighbors)
            hyper_features.append(hyper_feature)
            hyper_and_neighbors.append({'h_and_n': neighbors, 'h': hyper_node})

            # during subgraph BPE (merging stage),
            # keep track of the new pattern (most frequently appeared edge) and update the frequency on previous vocab
            if hyper_node_name not in self.com2sim_node_names['Round_{}'.format(round_n)].keys():
                self.num_of_hyper_node += 1
                hyper_node_name_simple = '{}R{}'.format(round_n, self.num_of_hyper_node)
                self.com2sim_node_names['Round_{}'.format(round_n)][hyper_node_name] = hyper_node_name_simple
                self.vocab_freq[hyper_node_name_simple] = 1
            else:
                # read the simple hyper-node name saved in the mapping
                hyper_node_name_simple = self.com2sim_node_names['Round_{}'.format(round_n)][hyper_node_name]
                self.vocab_freq[hyper_node_name_simple] += 1

            start_idx, end_idx = hyper_node
            start_node_name, end_node_name = node_names[start_idx], node_names[end_idx]
            self.vocab_freq[start_node_name] -= 1
            self.vocab_freq[end_node_name] -= 1

            hyper_split = node_split[start_idx] + node_split[end_idx]  # a list consists of origin node indices
            # node_split: [[node_idx], [node_idx1, node_idx2], ...]
            node_split.append(hyper_split)

            hyper_names.append(hyper_node_name_simple)  # previous: hyper_names.append(hyper_node_name)

            deleting_nodes.update(hyper_node)

            hyper_node_neighbors = np.nonzero(neighbors)[0]  # neighbors is a 1-d vector
            for j in range(i + 1, num_hyper_nodes):  # i+1, ..., num_hyper_nodes-1
                other_hyper_node = contracted_nodes[j]
                shared_nodes = set(hyper_node_neighbors) & set(other_hyper_node)
                if len(shared_nodes) > 0:
                    # for rings, len(shared_nodes) >=2; for non-rings, there shouldn't be shared nodes[wrong, see below]
                    # two hyper-nodes to merge lined by an extra edge  (H_node-H_node)
                    hyper2hyper[i, j] = 1

        hyper2normal = np.stack(hyper2normal, axis=0)  # (num_hyper_node, num_normal_node)

        hyper2hyper = hyper2hyper + hyper2hyper.T  # (num_hyper_node, num_hyper_node)

        augmented_adjacency = np.block([[adjacency_matrix, hyper2normal.T], [hyper2normal, hyper2hyper]])
        augmented_node_feature = np.concatenate([node_features, np.stack(hyper_features, axis=0)], axis=0)
        # (num_node, feature_dim) cat (num_hyper_node, feature_dim) ---> (num_node + num_hyper_node, feature_dim)
        node_names = node_names + hyper_names

        self.get_graph_grammar(hyper_and_neighbors=hyper_and_neighbors, num_hyper_nodes=num_hyper_nodes,
                               adj_len=adj_len,
                               aug_adj=augmented_adjacency, hyper2hyper_adj=hyper2hyper, aug_node_names=node_names,
                               round_n=round_n)

        deleting_nodes = list(deleting_nodes)

        augmented_adjacency = np.delete(augmented_adjacency, deleting_nodes, axis=0)
        augmented_adjacency = np.delete(augmented_adjacency, deleting_nodes, axis=1)
        augmented_node_feature = np.delete(augmented_node_feature, deleting_nodes, axis=0)
        node_split = np.delete(node_split, deleting_nodes, axis=0).tolist()

        # delete contracted nodes
        node_names = np.delete(node_names, deleting_nodes, axis=0).tolist()

        return augmented_adjacency, node_names, augmented_node_feature, node_split

    def contract_ring(self, adjacency_matrix: np.ndarray, contracted_nodes: list[list[int, ...]], node_names: list,
                      node_features: np.ndarray, node_split: list, consider_outside_neighbors=True):
        adj_len, node_names_len = len(adjacency_matrix), len(node_names)
        num_originally_connected_components = len(self.find_connected_components(adjacency_matrix=adjacency_matrix))
        assert adj_len == node_names_len, \
            'length of adjacency matrix ({}) does not match node names ({})'.format(adj_len, node_names_len)
        # update normal-node vocab (there are no hyper-node yet)
        if not self.contract_cliques:
            for node_name in node_names:
                if node_name not in self.vocab_freq.keys():
                    self.vocab_freq[node_name] = 1
                else:
                    self.vocab_freq[node_name] += 1
                self.vocab_origin.update(node_names)

        num_hyper_nodes = len(contracted_nodes)

        if num_hyper_nodes == 0:
            return adjacency_matrix, node_names, node_features, node_split  # (num_nodes, feature_dim)

        deleting_nodes = set()  # keep track of the nodes that constitute hyper-nodes

        # find the neighbors of each hyper_node
        hyper2normal = []  # stores adjacencies between hyper-nodes and normal-nodes
        hyper2hyper = np.zeros(shape=(num_hyper_nodes, num_hyper_nodes))
        # stores adjacencies between hyper-nodes and hyper-nodes
        whatif_hyper2hyper_sharing1node = np.zeros(shape=(num_hyper_nodes, num_hyper_nodes))
        hyper_names = []
        hyper_features = []  # len(hyper_features) = num_hyper_nodes
        hyper_and_neighbors = []

        for i in range(num_hyper_nodes):

            hyper_node = contracted_nodes[i]
            # during init, this can be a list consisting of multiple node indices (ring)
            # in other situations, it only contains two nodes

            hyper_node_components = []  # stores normal/original node indices
            for node_idx in hyper_node:  # the node idx for the (potentially contracted if contract clique) hyper-nodes
                hyper_node_components += node_split[node_idx]  # each hyper-node is consisted of a bunch of normal nodes
            node_split.append(hyper_node_components)

            neighbors, hyper_node_name, hyper_feature = self.get_hyper_node_neighbors_name_feature(
                adjacency_matrix=adjacency_matrix, hyper_node=hyper_node,
                node_names=node_names, node_features=node_features,
                consider_outside_neighbors=consider_outside_neighbors)
            # neighbors: 0-1 vectors indicating both the hyper nodes and the neighboring nodes in the adjacency matrix
            hyper2normal.append(neighbors)
            hyper_features.append(hyper_feature)  # sum of features for all the nodes within the hyper-node
            # record infor (hyper-node&neighbors, neighbors) for grammar learning
            hyper_and_neighbors.append({'h_and_n': neighbors, 'h': hyper_node})

            # update hyper-node vocab
            # during initialization, keep track of the variants of the same rings
            # (e.g., C-O-C [ring] is the same with O-C-C [ring])
            is_identical = False
            for existing_hyper in self.com2sim_node_names['Rings'].keys():
                is_identical = self.are_strings_identical(existing_str=existing_hyper, query=hyper_node_name)
                if is_identical:
                    hyper_node_name = existing_hyper
                    break
            if is_identical:
                # restore the simple version of hyper-node name from the mapping
                hyper_node_name_simple = self.com2sim_node_names['Rings'][hyper_node_name]
                self.vocab_freq[hyper_node_name_simple] += 1
            else:
                self.num_of_hyper_node += 1
                # store the simple version of hyper-node name in the mapping
                hyper_node_name_simple = '{}R{}'.format('*', self.num_of_hyper_node)
                self.com2sim_node_names['Rings'][hyper_node_name] = hyper_node_name_simple
                self.vocab_freq[hyper_node_name_simple] = 1
                # put new hyper-node string in the vocab

            hyper_names.append(hyper_node_name_simple)

            deleting_nodes.update(hyper_node)

            hyper_node_neighbors = np.nonzero(neighbors)[0]  # neighbors is a 1-d vector
            for j in range(i + 1, num_hyper_nodes):  # i+1, ..., num_hyper_nodes-1
                other_hyper_node = contracted_nodes[j]
                shared_nodes = set(hyper_node) & set(other_hyper_node)
                shared_nodes_w_neighbors = set(hyper_node_neighbors) & set(other_hyper_node)

                if len(shared_nodes) >= 1:  # shared edge(>=2)
                    hyper2hyper[i, j] = 1
                    hyper2hyper[j, i] = 1
                if len(shared_nodes) == 1:
                    whatif_hyper2hyper_sharing1node[i, j] = 1
                    whatif_hyper2hyper_sharing1node[j, i] = 1
                    # there is only one shared node between two cliques (however we need to make sure node-i
                    # has no shared edges with the rest nodes)
                    is_sharing_edge_with_third_ring = bool(
                        (hyper2hyper[i] > 1).any())  # check the previous nodes (before j)
                    if not is_sharing_edge_with_third_ring:
                        for k in range(j + 1, num_hyper_nodes):  # check the rest nodes (after j)
                            third_hyper_node = contracted_nodes[k]
                            shared_nodes_with_third_ring = set(hyper_node) & set(third_hyper_node)
                            if len(shared_nodes_with_third_ring) > 1:
                                is_sharing_edge_with_third_ring = True
                                break
                    if not is_sharing_edge_with_third_ring:
                        hyper2hyper[i, j] = 1
                        hyper2hyper[j, i] = 1

                if len(shared_nodes_w_neighbors) == 1:
                    # some example: two triangles linked by an extra edge (we need to make sure this extra edge is not
                    # in a third ring); two hyper-nodes to merge lined by an extra edge (H_node-H_node)
                    is_in_third_ring = False
                    shared_node_in_j = list(shared_nodes_w_neighbors)[0]
                    shared_node_neighbors = np.nonzero(adjacency_matrix[shared_node_in_j])[0].tolist()
                    neighbors_of_shared_node_in_hyper_node_i = set(shared_node_neighbors) & set(hyper_node)
                    assert len(neighbors_of_shared_node_in_hyper_node_i) >= 1
                    if len(neighbors_of_shared_node_in_hyper_node_i) == 1:
                        # if > 1, it means there are other cliques that will connect i, j
                        edge_node_in_i = list(neighbors_of_shared_node_in_hyper_node_i)[0]
                        for k in range(num_hyper_nodes):
                            if k == i or k == j:  # we don't consider the current node i, j
                                continue
                            third_hyper_node = contracted_nodes[k]
                            if shared_node_in_j in third_hyper_node and edge_node_in_i in third_hyper_node:
                                # means the extra edge between the i-th current hyper-node and the j-th other hyper-node
                                # is in the k-th third hyper-node; thus, don't connect i, j (since k will connect i, j)
                                is_in_third_ring = True
                                break
                        if not is_in_third_ring:
                            hyper2hyper[i, j] = 1
                            hyper2hyper[j, i] = 1

        hyper2normal = np.stack(hyper2normal, axis=0)  # (num_hyper_node, num_normal_node)

        assert np.allclose(2 * hyper2hyper, hyper2hyper + hyper2hyper.T)  # assert symmetry

        augmented_adjacency = np.block([[adjacency_matrix, hyper2normal.T], [hyper2normal, hyper2hyper]])
        augmented_node_feature = np.concatenate([node_features, np.stack(hyper_features, axis=0)], axis=0)
        # (num_node + num_hyper_node, feature_dim)
        node_names = node_names + hyper_names

        self.get_graph_grammar(hyper_and_neighbors=hyper_and_neighbors, num_hyper_nodes=num_hyper_nodes,
                               adj_len=adj_len,
                               aug_adj=augmented_adjacency, hyper2hyper_adj=hyper2hyper, aug_node_names=node_names,
                               round_n=0)

        deleting_nodes = list(deleting_nodes)

        for node_idx in deleting_nodes:
            # update the frequency on the deleted nodes
            node_name = node_names[node_idx]
            self.vocab_freq[node_name] -= 1

        augmented_adjacency = np.delete(augmented_adjacency, deleting_nodes, axis=0)
        augmented_adjacency = np.delete(augmented_adjacency, deleting_nodes, axis=1)

        augmented_adjacency = self.make_connected_topology(augmented_adjacency=augmented_adjacency,
                                                           hyper2hyper=hyper2hyper,
                                                           whatif_hyper2hyper_sharing1node=whatif_hyper2hyper_sharing1node,
                                                           num_hyper_nodes=num_hyper_nodes,
                                                           num_originally_connected_components=num_originally_connected_components)

        augmented_node_feature = np.delete(augmented_node_feature, deleting_nodes, axis=0)
        node_split = np.delete(node_split, deleting_nodes, axis=0).tolist()
        # delete contracted nodes
        node_names = np.delete(node_names, deleting_nodes, axis=0).tolist()

        return augmented_adjacency, node_names, augmented_node_feature, node_split

    def contract_clique(self, adjacency_matrix: np.ndarray, contracted_nodes: list[list[int, ...]], node_names: list,
                        node_features: np.ndarray, node_split: list[list[int, ...]], consider_outside_neighbors=False):
        adj_len, node_names_len = len(adjacency_matrix), len(node_names)
        assert adj_len == node_names_len, \
            'length of adjacency matrix ({}) does not match node names ({})'.format(adj_len, node_names_len)

        num_originally_connected_components = len(self.find_connected_components(adjacency_matrix=adjacency_matrix))
        # update normal-node vocab (there are no hyper-node yet)
        for node_name in node_names:
            if node_name not in self.vocab_freq.keys():
                self.vocab_freq[node_name] = 1
            else:
                self.vocab_freq[node_name] += 1
            self.vocab_origin.update(node_names)

        num_hyper_nodes = len(contracted_nodes)

        if num_hyper_nodes == 0:
            return adjacency_matrix, node_names, node_features, node_split  # (num_nodes, feature_dim)

        deleting_nodes = set()  # keep track of the nodes that constitute hyper-nodes

        # find the neighbors of each hyper_node
        hyper2normal = []  # stores adjacencies between hyper-nodes and normal-nodes
        hyper2hyper = np.zeros(shape=(num_hyper_nodes, num_hyper_nodes))
        # stores adjacencies between hyper-nodes and hyper-nodes
        whatif_hyper2hyper_sharing1node = np.zeros(shape=(num_hyper_nodes, num_hyper_nodes))
        # key (hyper-node idx): value (neighbors for that hyper-node if we connect when there is exactly 1 shared node)
        whatif_hyper2hyper_sharing_more_than_2_neighbors = np.zeros(shape=(num_hyper_nodes, num_hyper_nodes))
        hyper_names = []
        hyper_features = []  # len(hyper_features) = num_hyper_nodes
        hyper_and_neighbors = []

        for i in range(num_hyper_nodes):
            # neighbors = adjacency_matrix[hyper_node].sum(axis=0)
            # neighbors[neighbors > 0] = 1

            hyper_node = contracted_nodes[i]
            # during init, this can be a list consisting of multiple node indices (ring)
            # in other situations, it only contains two nodes

            hyper_node_components = []  # stores normal/original node indices
            for node_idx in hyper_node:
                hyper_node_components += node_split[node_idx]
            node_split.append(hyper_node_components)

            neighbors, hyper_node_name, hyper_feature = self.get_hyper_node_neighbors_name_feature(
                adjacency_matrix=adjacency_matrix, hyper_node=hyper_node,
                node_names=node_names, node_features=node_features,
                consider_outside_neighbors=consider_outside_neighbors)
            # neighbors: 0-1 vectors indicating both the hyper nodes and the neighboring nodes in the adjacency matrix
            hyper2normal.append(neighbors)
            hyper_features.append(hyper_feature)  # sum of features for all the nodes within the hyper-node
            # record infor (hyper-node&neighbors, neighbors) for grammar learning
            hyper_and_neighbors.append({'h_and_n': neighbors, 'h': hyper_node})

            # update hyper-node vocab
            # during initialization, keep track of the variants of the same rings
            # (e.g., C-O-C [ring] is the same with O-C-C [ring])
            is_identical = False
            for existing_hyper in self.com2sim_node_names['Cliques'].keys():
                is_identical = self.are_strings_identical(existing_str=existing_hyper, query=hyper_node_name)
                if is_identical:
                    hyper_node_name = existing_hyper
                    break
            if is_identical:
                # restore the simple version of hyper-node name from the mapping
                hyper_node_name_simple = self.com2sim_node_names['Cliques'][hyper_node_name]
                self.vocab_freq[hyper_node_name_simple] += 1
            else:
                self.num_of_hyper_node += 1
                # store the simple version of hyper-node name in the mapping
                hyper_node_name_simple = '{}C{}'.format('*', self.num_of_hyper_node)
                self.com2sim_node_names['Cliques'][hyper_node_name] = hyper_node_name_simple
                self.vocab_freq[hyper_node_name_simple] = 1
                # put new hyper-node string in the vocab

            hyper_names.append(hyper_node_name_simple)

            deleting_nodes.update(hyper_node)

            hyper_node_neighbors = np.nonzero(neighbors)[0]  # neighbors is a 1-d vector
            for j in range(i + 1, num_hyper_nodes):  # i+1, ..., num_hyper_nodes-1
                other_hyper_node = contracted_nodes[j]
                shared_nodes = set(hyper_node) & set(other_hyper_node)
                shared_nodes_w_neighbors = set(hyper_node_neighbors) & set(other_hyper_node)

                if len(shared_nodes) > 1:  # shared edge
                    hyper2hyper[i, j] = len(shared_nodes)
                    hyper2hyper[j, i] = len(shared_nodes)
                if len(shared_nodes) == 1:
                    whatif_hyper2hyper_sharing1node[i, j] = 1
                    whatif_hyper2hyper_sharing1node[j, i] = 1
                    # there is only one shared node between two cliques (however we need to make sure node-i
                    # has no shared edges with the rest nodes)
                    is_sharing_edge_with_third_clique = bool(
                        (hyper2hyper[i] > 1).any())  # check the previous nodes (before j)
                    if not is_sharing_edge_with_third_clique:
                        for k in range(j + 1, num_hyper_nodes):  # check the rest nodes (after j)
                            third_hyper_node = contracted_nodes[k]
                            shared_nodes_with_third_clique = set(hyper_node) & set(third_hyper_node)
                            if len(shared_nodes_with_third_clique) > 1:
                                is_sharing_edge_with_third_clique = True
                                break
                    if not is_sharing_edge_with_third_clique:
                        hyper2hyper[i, j] = 1
                        hyper2hyper[j, i] = 1

                if len(shared_nodes_w_neighbors) == 1:
                    # some example: two triangles linked by an extra edge (we need to make sure this extra edge is not
                    # in a third ring); two hyper-nodes to merge lined by an extra edge (H_node-H_node)
                    is_in_third_clique = False
                    shared_node_in_j = list(shared_nodes_w_neighbors)[0]
                    shared_node_neighbors = np.nonzero(adjacency_matrix[shared_node_in_j])[0].tolist()
                    neighbors_of_shared_node_in_hyper_node_i = set(shared_node_neighbors) & set(hyper_node)
                    assert len(neighbors_of_shared_node_in_hyper_node_i) >= 1
                    if len(neighbors_of_shared_node_in_hyper_node_i) == 1:
                        # if > 1, it means there are other cliques that will connect i, j
                        edge_node_in_i = list(neighbors_of_shared_node_in_hyper_node_i)[0]
                        for k in range(num_hyper_nodes):
                            if k == i or k == j:  # we don't consider the current node i, j
                                continue
                            third_hyper_node = contracted_nodes[k]
                            if shared_node_in_j in third_hyper_node and edge_node_in_i in third_hyper_node:
                                # means the extra edge between the i-th current hyper-node and the j-th other hyper-node
                                # is in the k-th third hyper-node; thus, don't connect i, j (since k will connect i, j)
                                is_in_third_clique = True
                                break
                        if not is_in_third_clique:
                            hyper2hyper[i, j] = 1
                            hyper2hyper[j, i] = 1

                if len(shared_nodes_w_neighbors) >= 2:  # for cliques, it's possible that there are multiple bridges,
                    # e.g., two 4-cliques connected by two parallel edges
                    whatif_hyper2hyper_sharing_more_than_2_neighbors[i, j] = 1
                    whatif_hyper2hyper_sharing_more_than_2_neighbors[j, i] = 1

        hyper2normal = np.stack(hyper2normal, axis=0)  # (num_hyper_node, num_normal_node)

        hyper2hyper[hyper2hyper > 0] = 1
        assert np.allclose(2 * hyper2hyper, hyper2hyper + hyper2hyper.T)  # assert symmetry

        augmented_adjacency = np.block([[adjacency_matrix, hyper2normal.T], [hyper2normal, hyper2hyper]])

        augmented_node_feature = np.concatenate([node_features, np.stack(hyper_features, axis=0)], axis=0)
        # (num_node + num_hyper_node, feature_dim)
        node_names = node_names + hyper_names

        self.get_graph_grammar(hyper_and_neighbors=hyper_and_neighbors, num_hyper_nodes=num_hyper_nodes,
                               adj_len=adj_len,
                               aug_adj=augmented_adjacency, hyper2hyper_adj=hyper2hyper, aug_node_names=node_names,
                               round_n='c')

        deleting_nodes = list(deleting_nodes)

        for node_idx in deleting_nodes:
            # update the frequency on the deleted nodes
            node_name = node_names[node_idx]
            self.vocab_freq[node_name] -= 1

        augmented_adjacency = np.delete(augmented_adjacency, deleting_nodes, axis=0)
        augmented_adjacency = np.delete(augmented_adjacency, deleting_nodes, axis=1)

        augmented_adjacency = self.make_connected_topology(augmented_adjacency=augmented_adjacency,
                                                           hyper2hyper=hyper2hyper,
                                                           whatif_hyper2hyper_sharing1node=whatif_hyper2hyper_sharing1node,
                                                           num_hyper_nodes=num_hyper_nodes,
                                                           whatif_hyper2hyper_sharing_more_than_2_neighbors=whatif_hyper2hyper_sharing_more_than_2_neighbors,
                                                           num_originally_connected_components=num_originally_connected_components)

        augmented_node_feature = np.delete(augmented_node_feature, deleting_nodes, axis=0)
        node_split = np.delete(node_split, deleting_nodes, axis=0).tolist()
        # delete contracted nodes
        node_names = np.delete(node_names, deleting_nodes, axis=0).tolist()

        return augmented_adjacency, node_names, augmented_node_feature, node_split

    def make_connected_topology(self, augmented_adjacency, hyper2hyper, whatif_hyper2hyper_sharing1node,
                                num_hyper_nodes, num_originally_connected_components,
                                whatif_hyper2hyper_sharing_more_than_2_neighbors=None):
        if len(self.find_connected_components(
                adjacency_matrix=augmented_adjacency)) == num_originally_connected_components:
            # already connected (via hyper2normal connections), do nothing
            return augmented_adjacency
        else:
            # need to make the hyper2hyper connected (since we ignored certain connections for len(shared_nodes) == 1)
            connected_components = self.find_connected_components(adjacency_matrix=hyper2hyper)  # list[set]
            assert len(connected_components) > 1
            for component in connected_components:
                component = list(component)
                for node in component:
                    whatif_hyper2hyper_sharing1node[node, component] = 0  # remove in-cluster connections
            hyper2hyper = hyper2hyper + whatif_hyper2hyper_sharing1node
            # add inter-cluster connections if not existing in the original hyper2hyper topology
            hyper2hyper[hyper2hyper > 0] = 1
            augmented_adjacency[-num_hyper_nodes:, -num_hyper_nodes:] = hyper2hyper

            def make_connected_topology_again(augmented_adjacency, hyper2hyper, num_hyper_nodes,
                                              whatif_hyper2hyper_sharing_more_than_2_neighbors,
                                              num_originally_connected_components):
                num_connected = len(self.find_connected_components(adjacency_matrix=augmented_adjacency))
                if whatif_hyper2hyper_sharing_more_than_2_neighbors is None:
                    assert num_connected == num_originally_connected_components
                    return augmented_adjacency
                else:  # there are special cases that the topology is still unconnected (e.g., a pair of parallel edges
                    # connect two 4-cliques)
                    if num_connected == num_originally_connected_components:
                        return augmented_adjacency
                    else:
                        connected_cluster = set()
                        connected_components = self.find_connected_components(adjacency_matrix=hyper2hyper)
                        num_connected_for_multiple_bridges = len(connected_components)
                        for i in range(num_connected_for_multiple_bridges):
                            component_i = connected_components[i]
                            for j in range(i + 1, num_connected_for_multiple_bridges):
                                if {i, j}.issubset(connected_cluster):  # cluster_i/j is already made connected
                                    continue
                                component_j = connected_components[j]
                                is_connected, hyper2hyper = self.find_connecting_edge(component_i=component_i,
                                                                                      component_j=component_j,
                                                                                      hyper2hyper=hyper2hyper,
                                                                                      whatif_hyper2hyper_sharing_more_than_2_neighbors=whatif_hyper2hyper_sharing_more_than_2_neighbors)
                                if is_connected:
                                    connected_cluster.update([i, j])
                        augmented_adjacency[-num_hyper_nodes:, -num_hyper_nodes:] = hyper2hyper
                        # there exists cases (e.g., enzymes_52) that originally contain two or more disjoint graphs
                        assert len(self.find_connected_components(
                            adjacency_matrix=augmented_adjacency)) == num_originally_connected_components
                        return augmented_adjacency

            augmented_adjacency = make_connected_topology_again(augmented_adjacency=augmented_adjacency,
                                                                hyper2hyper=hyper2hyper,
                                                                num_hyper_nodes=num_hyper_nodes,
                                                                whatif_hyper2hyper_sharing_more_than_2_neighbors=whatif_hyper2hyper_sharing_more_than_2_neighbors,
                                                                num_originally_connected_components=num_originally_connected_components)
            return augmented_adjacency

    @staticmethod
    def find_connecting_edge(component_i, component_j, hyper2hyper, whatif_hyper2hyper_sharing_more_than_2_neighbors):
        # connect the two components by add potential connecting edges between them
        is_connected = False
        for node_idx in component_i:
            node_neighbors = np.nonzero(whatif_hyper2hyper_sharing_more_than_2_neighbors[node_idx])[0]
            for neighbor_idx in node_neighbors:
                if neighbor_idx in component_j:
                    is_connected = True
                    assert hyper2hyper[node_idx, neighbor_idx] == 0
                    hyper2hyper[node_idx, neighbor_idx] = 1
                    hyper2hyper[neighbor_idx, node_idx] = 1
                    return is_connected, hyper2hyper
        return is_connected, hyper2hyper

    @staticmethod
    def find_connected_components(adjacency_matrix: np.ndarray):
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(adjacency_matrix)
        # Find connected components
        connected_components = list(nx.connected_components(G))

        return connected_components  # list[set]

    def get_graph_grammar(self, hyper_and_neighbors: list[dict], num_hyper_nodes: int, adj_len: int,
                          aug_adj: np.ndarray, hyper2hyper_adj: np.ndarray, aug_node_names: list[str], round_n):

        connected_hyper_nodes = self.find_connected_components(adjacency_matrix=hyper2hyper_adj)
        for connected_component in connected_hyper_nodes:
            before_merge_nodes = np.zeros(shape=(adj_len,))
            hyper_nodes_idx = []
            for idx in connected_component:  # a set
                hyper_node = hyper_and_neighbors[idx]
                h_n, h = hyper_node['h_and_n'], hyper_node['h']
                before_merge_nodes += h_n
                hyper_nodes_idx += h

            # create before merge topology, node names
            before_merge_idx = np.nonzero(before_merge_nodes)[0]
            # idx within the range of normal node, indicating hyper-nodes and their neighbors
            before_merge_adj = aug_adj[before_merge_idx]
            before_merge_adj = before_merge_adj[:, before_merge_idx]
            before_merge_node_names = np.take(aug_node_names, before_merge_idx).tolist()

            # create after merge topology, node names
            hyper_nodes_idx = list(set(hyper_nodes_idx))
            before_merge_nodes[hyper_nodes_idx] = 0  # set the hyper-nodes idx (within the range of normal-node) to 0
            connected_component_tensor = np.zeros(shape=(num_hyper_nodes,))
            connected_component_tensor[list(connected_component)] = 1
            after_merge_nodes = np.concatenate([before_merge_nodes, connected_component_tensor], axis=0)
            # num_normal_nodes (--cat--) num_hyper_nodes ---> num_normal_nodes + num_hyper_nodes
            after_merge_idx = np.nonzero(after_merge_nodes)[0]
            after_merge_adj = aug_adj[after_merge_idx]
            after_merge_adj = after_merge_adj[:, after_merge_idx]
            after_merge_node_names = np.take(aug_node_names, after_merge_idx).tolist()

            graph_grammar = {'LHS': {'adj': after_merge_adj, 'node_name': after_merge_node_names},
                             'RHS': {'adj': before_merge_adj, 'node_name': before_merge_node_names}}

            self.graph_grammar[round_n].append(graph_grammar)

    @staticmethod
    def get_hyper_node_neighbors_name_feature(adjacency_matrix: np.ndarray, hyper_node: list[int],
                                              node_names: list[str], node_features: np.ndarray,
                                              consider_outside_neighbors=True):
        neighbors = []
        hyper_node_name = []
        hyper_feature = np.zeros(shape=(node_features.shape[1],))
        for node_idx in hyper_node:  # node is an integer
            hyper_feature += node_features[node_idx]
            # find neighbors
            node_neighbor = adjacency_matrix[node_idx]
            neighbors.append(node_neighbor)

            # get hyper-node name
            if consider_outside_neighbors:
                outside_neighbor_name = []  # neighboring nodes that are not in the hyper-node
                for n in np.nonzero(node_neighbor)[0]:  # 0 because node_neighbor is a 1-d array
                    if n not in hyper_node:
                        outside_neighbor_name.append(node_names[n])  # node_names[n] is a string
                outside_neighbor_name.sort()  # we don't want orderings on the outside neighbors

                node_name = node_names[node_idx] + '[' + ''.join(outside_neighbor_name) + ']'
            else:
                node_name = node_names[node_idx]

            hyper_node_name.append(node_name)

        hyper_node_name.sort()  # 'C=0' should be treated the same as 'O=C'
        # find neighbors
        neighbors = sum(neighbors)  # [(num_node, )] x num_hyper_node ---> (num_node, )
        # after sum, all the neighboring nodes would have none zero entries,
        # it's ok the neighbors contain nodes in hyper_node, since they will be deleted later

        neighbors[neighbors > 0] = 1

        # get hyper-node name
        hyper_node_name = '(' + ''.join(hyper_node_name) + ')'
        return neighbors, hyper_node_name, hyper_feature

    @staticmethod
    def are_strings_identical(existing_str, query):
        if len(existing_str) != len(query):
            return False

        existing_str = existing_str[1:-1]  # remove the '(' and ')' token
        query = query[1:-1]  # remove the '(' and ')' token

        concatenated_str = existing_str + existing_str
        return query in concatenated_str

    def get_pairs_and_count(self, adjacency_matrix, node_names, mol_name):
        # update co-occurrences
        num_node = len(adjacency_matrix)
        for i in range(num_node):
            for j in range(i + 1, num_node):
                if adjacency_matrix[i][j] == 1:
                    name = [node_names[i], node_names[j]]
                    name.sort()  # 'C=0' should be treated the same as 'O=C'
                    name = '-'.join(name)
                    if name in self.co_occurrence.keys():
                        self.co_occurrence[name]['freq'] += 1
                        # 'name' has been spotted before
                        if mol_name not in self.co_occurrence[name]['mols'].keys():
                            # 'name' is spotted the first time in 'mol_name'
                            self.co_occurrence[name]['mols'][mol_name] = [[i, j]]
                        else:
                            # 'name' has been spotted in 'mol_name' before
                            self.co_occurrence[name]['mols'][mol_name].append([i, j])
                    else:
                        self.co_occurrence[name] = {'freq': 1, 'mols': {mol_name: [[i, j]]}}
                        # 'name' is spotted the first time


@hydra.main(version_base='1.3', config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    subgraph_tokenizer = GraphBPE(dataset_cfg=cfg.dataset)
    subgraph_tokenizer.process()


if __name__ == '__main__':
    main()