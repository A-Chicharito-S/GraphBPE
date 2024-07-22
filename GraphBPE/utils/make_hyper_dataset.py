import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import torch


# make dataset in the centroid way, as introduced in the HGNN paper
def make_centroid_hypergraph(dataset):
    hyper_dataset = []
    for data in dataset:
        adj = nx.adjacency_matrix(to_networkx(data, to_undirected=True)).todense()
        num_nodes = len(adj)
        hyperedge_index = torch.eye(n=num_nodes) + torch.from_numpy(adj)
        # add the node itself and add the neighbors (defined by the adj matrix)
        hyperedge_index = torch.nonzero(hyperedge_index).T
        data.hyperedge_index = hyperedge_index
        hyper_dataset.append(data)
    return hyper_dataset


def get_hyperedge_index(edge_index: np.ndarray, node_split: list):
    # by definition, each hyper-edge connects >= 2 vertices https://arxiv.org/pdf/1901.08150.pdf
    if edge_index.shape[1] == 0:
        # e.g., Data(x=[1, 9], edge_index=[2, 0], edge_attr=[0, 3], y=[1, 1], smiles='C', node_names=[1], name='esol_935')
        hyperedge_index = np.zeros(shape=(2, 1), dtype=edge_index.dtype)
        return torch.from_numpy(hyperedge_index)
    row_idx, col_idx = edge_index
    num_nodes = max(row_idx) + 1
    edge_flag = np.less(row_idx, col_idx)
    # e.g., keep edge (1, 0) [marked as False] and remove edge (0, 1) [marked as True]
    edge_index = np.delete(edge_index, edge_flag, axis=1)  # (2, num_edges)
    row_idx, col_idx = edge_index
    num_edges = len(row_idx)

    incidence_matrix = np.zeros(shape=(num_nodes, num_edges))  # stores the original edges
    incidence_matrix[row_idx, np.arange(num_edges)] = 1  # row_idx represents the start node of some edge
    incidence_matrix[col_idx, np.arange(num_edges)] = 1  # col_idx represents the end node of that edge

    hyper_node_split = [split for split in node_split if len(split) > 1]  # if len(split)==1, the node is not merged
    deleting_edges = set()
    for edge_idx in range(num_edges):
        node1, node2 = row_idx[edge_idx], col_idx[edge_idx]
        # we don't distinguish start/end node here (as the graph is undirected)
        for hyper_merge in hyper_node_split:
            if node1 in hyper_merge and node2 in hyper_merge:
                deleting_edges.add(edge_idx)
                break  # even if such an edge exists in multiple hyper-nodes (e.g., in the case of the shared edge
                # of two rings), finding its existence in one hyper-node is sufficient (to delete them)

    # construct incidence matrix for hyper-nodes
    auxiliary_incidence = np.zeros(shape=(num_nodes, len(hyper_node_split)))
    for edge_idx, hyper_merge in enumerate(hyper_node_split):
        for node_idx in hyper_merge:
            auxiliary_incidence[node_idx][edge_idx] = 1

    augmented_incidence = np.concatenate([incidence_matrix, auxiliary_incidence], axis=1)
    # (num_nodes, num_edges) cat (num_nodes, num_hyper_edges) ---> (num_nodes, num_edges+num_hyper_edges)
    augmented_incidence = np.delete(augmented_incidence, list(deleting_edges), axis=1)

    hyperedge_row, hyperedge_col = np.nonzero(augmented_incidence)
    hyperedge_index = np.array([hyperedge_row, hyperedge_col])

    return torch.from_numpy(hyperedge_index)
