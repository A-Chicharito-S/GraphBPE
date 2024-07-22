from torch_geometric.datasets.tu_dataset import *
from torch_geometric.data import Data


def pre_transform_enzymes(idx: int, data: Data):
    idx2node = {0: 'E1', 1: 'E2', 2: 'E3'}
    x = data.x[:, -3:]  # 3 = |idx2node|, features of x = [node_features(if present) ||(concat) one-hot node label]
    # TODO: above behavior is defined by read_tu_data; however, with package upgrades, such a behavior might change
    assert torch.equal(x.sum(-1), torch.ones(size=(x.shape[0], )))
    # simple assertion on whether x is categorical/one-hot
    x_labels = torch.nonzero(x).tolist()  # (num_nodes, 2[row_idx, column_idx]
    node_names = [idx2node[idx[-1]] for idx in x_labels]
    data.node_names = node_names
    data.name = 'enzymes_' + str(idx + 1)
    return data


class AugmentedEnzymes(TUDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        assert name == 'ENZYMES'
        super().__init__(root=root, name=name, transform=transform, pre_transform=pre_transform_enzymes,
                         pre_filter=pre_filter, use_node_attr=use_node_attr, cleaned=cleaned)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(idx=idx, data=d) for idx, d in enumerate(data_list)]
                # add name to the Data object

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self._data.to_dict(), self.slices, sizes),
                   self.processed_paths[0])
