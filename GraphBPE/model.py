import pytorch_lightning as pl
from torch import optim, nn
from torchmetrics.classification import BinaryAccuracy, Accuracy
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
import torch
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, HypergraphConv, GCN, GAT, GIN, GraphSAGE, \
    TransformerConv
import torch.nn.functional as F
from supported import supported_model
from torch_geometric.nn.norm import GraphNorm, BatchNorm
from dhg.nn import HGNNConv, HGNNPConv, HNHNConv
from dhg import Hypergraph
from collections import defaultdict


class WeightChangeTracker:
    """
    Check if the weights of NN change
    """

    def __init__(self, model, model_name):
        super(WeightChangeTracker, self).__init__()
        self.model = model
        self.model_name = model_name
        assert isinstance(self.model, nn.Module)
        self.weights = {}
        self._store_weights()

    def _store_weights(self):
        for name, param in self.model.named_parameters():
            self.weights[name] = param.clone().detach()

    def assert_change(self):
        for name, param in self.model.named_parameters():
            if self.weights[name].to(param.device).equal(param):
                continue
            print('\"{}\" weights changed: {}'.format(self.model_name, name))


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads, norm):
        super(GraphTransformer, self).__init__()
        self.num_layer = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels, heads=heads))
            else:
                layers.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=heads))
        self.layers = nn.ModuleList(layers)

        if norm is None:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(num_layers - 1)])
        elif norm == 'BatchNorm':
            self.norms = nn.ModuleList([BatchNorm(in_channels=hidden_channels) for _ in range(num_layers - 1)])
        elif norm == 'GraphNorm':
            self.norms = nn.ModuleList([GraphNorm(in_channels=hidden_channels) for _ in range(num_layers - 1)])
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x=x, edge_index=edge_index)
            if i != self.num_layer - 1:
                x = self.norms[i](x)
                x = x.relu()
            # to mimic the behavior of GCN, GIN, etc. of PyG
        return x


class HyperConvNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, norm):
        super(HyperConvNN, self).__init__()
        self.num_layer = num_layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = HypergraphConv(in_channels=in_channels, out_channels=hidden_channels)

            else:
                layer = HypergraphConv(in_channels=hidden_channels, out_channels=hidden_channels)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if norm is None:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        elif norm == 'BatchNorm':
            self.norms = nn.ModuleList([BatchNorm(in_channels=hidden_channels) for _ in range(num_layers)])
        elif norm == 'GraphNorm':
            self.norms = nn.ModuleList([GraphNorm(in_channels=hidden_channels) for _ in range(num_layers)])
        else:
            raise NotImplementedError

    def forward(self, x, hyperedge_index):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x=x, hyperedge_index=hyperedge_index)
            x = norm(x)
            x = x.relu()
            # to mimic the behavior of GCN, etc. of PyG (norm-then-activate),
            # and put all the normalization/activation inside
        return x


def get_hg(num_vertices: int, hyperedge_index: torch.Tensor):
    row_idx, col_idx = hyperedge_index
    assert len(row_idx) == len(col_idx), "row_index and col_index must have the same length."
    # Use a dictionary to group indices by the col_index value
    grouped = defaultdict(list)

    # Iterate over row_index and col_index to populate the dictionary
    for r, c in zip(row_idx, col_idx):
        grouped[c.item()].append(r.item())  # Convert tensors to native Python types

    # Return the grouped data as a list of lists
    edge_list = [grouped[key] for key in grouped.keys()]

    hg = Hypergraph(num_v=num_vertices, e_list=edge_list)
    hg = hg.to(hyperedge_index.device)
    return hg


class dhgHGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, norm):
        super(dhgHGNN, self).__init__()
        self.num_layer = num_layers

        if norm is None:
            use_bn = False
        elif norm == 'BatchNorm':
            use_bn = True
        else:
            raise NotImplementedError

        assert in_channels != -1  # lazy initialization is not supported
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = HGNNConv(in_channels=in_channels, out_channels=hidden_channels, use_bn=use_bn)
            else:
                layer = HGNNConv(in_channels=hidden_channels, out_channels=hidden_channels, use_bn=use_bn)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x, hyperedge_index):
        r"""The forward function.

                Args:
                    ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
                    ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
                """
        hg = get_hg(num_vertices=x.shape[0], hyperedge_index=hyperedge_index)
        for layer in self.layers:
            x = layer(x, hg)
        return x


class dhgHGNNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, norm):
        super(dhgHGNNP, self).__init__()
        self.num_layer = num_layers

        if norm is None:
            use_bn = False
        elif norm == 'BatchNorm':
            use_bn = True
        else:
            raise NotImplementedError

        assert in_channels != -1  # lazy initialization is not supported
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = HGNNPConv(in_channels=in_channels, out_channels=hidden_channels, use_bn=use_bn)
            else:
                layer = HGNNPConv(in_channels=hidden_channels, out_channels=hidden_channels, use_bn=use_bn)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x, hyperedge_index):
        r"""The forward function.

                Args:
                    ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
                    ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
                """
        hg = get_hg(num_vertices=x.shape[0], hyperedge_index=hyperedge_index)
        for layer in self.layers:
            x = layer(x, hg)
        return x


class dhgHNHN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, norm):
        super(dhgHNHN, self).__init__()
        self.num_layer = num_layers

        if norm is None:
            use_bn = False
        elif norm == 'BatchNorm':
            use_bn = True
        else:
            raise NotImplementedError

        assert in_channels != -1  # lazy initialization is not supported
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = HNHNConv(in_channels=in_channels, out_channels=hidden_channels, use_bn=use_bn)
            else:
                layer = HNHNConv(in_channels=hidden_channels, out_channels=hidden_channels, use_bn=use_bn)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x, hyperedge_index):
        r"""The forward function.

                Args:
                    ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
                    ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
                """
        hg = get_hg(num_vertices=x.shape[0], hyperedge_index=hyperedge_index)
        for layer in self.layers:
            x = layer(x, hg)
        return x


class CustomizedGNN(torch.nn.Module):
    def __init__(self, cfg, num_head=4):
        super(CustomizedGNN, self).__init__()

        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train

        model_id = self.model_cfg.model_id
        in_channels = self.model_cfg.in_channels
        hidden_channels = self.model_cfg.hidden_channels
        num_layer = self.model_cfg.num_layer
        num_class = self.dataset_cfg.num_class
        norm = self.train_cfg.norm

        self.dropout_rate = self.train_cfg.dropout_rate

        # self.layers = []
        # for i in range(num_layer):
        #     if i == 0:
        #         if model_id == 'GCN':
        #             layer = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        #             # print('called')
        #         elif model_id == 'GAT':
        #             layer = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=num_head)
        #         else:
        #             raise NotImplementedError
        #     else:
        #         if model_id == 'GCN':
        #             layer = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)
        #         elif model_id == 'GAT':
        #             layer = GATConv(in_channels=hidden_channels * num_head, out_channels=hidden_channels,
        #                             heads=num_head)
        #         else:
        #             raise NotImplementedError
        #     self.layers.append(layer)
        #
        # self.layers = nn.ModuleList(self.layers)
        #
        # self.lin = nn.Linear(in_features=hidden_channels * num_head, out_features=num_class) if model_id == 'GAT' \
        #     else nn.Linear(in_features=hidden_channels, out_features=num_class)

        if model_id == 'GCN':
            self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer, norm=norm)
            # in_channels ---> hidden_channels
        elif model_id == 'GAT':
            self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer,
                           heads=num_head, norm=norm)
            # in_channels ---> (hidden_channels // num_head) [==>(intermediate) out_dim] * num_head = hidden_channels
            # another way is to set hidden_channels = hidden_channels * num_head
        elif model_id == 'GIN':
            self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer, norm=norm)
        elif model_id == 'GraphSAGE':
            self.gnn = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer,
                                 norm=norm)
        elif model_id == 'GraphTransformer':
            self.gnn = GraphTransformer(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer,
                                        heads=num_head, norm=norm)
        else:
            raise NotImplementedError
        # call stack for above PyG models: (assuming N layers)
        # (1 layer of gnn + 1 layer of normalization + activation) x (N-1) + 1 layer of gnn

        if norm is None:
            self.norm = nn.Identity()
        elif norm == 'BatchNorm':
            self.norm = BatchNorm(in_channels=hidden_channels)
        elif norm == 'GraphNorm':
            self.norm = GraphNorm(in_channels=hidden_channels)
        else:
            raise NotImplementedError
        self.lin = nn.Linear(in_features=hidden_channels, out_features=num_class)
        # another way is to set in_features = hidden_channels * num_head for GAT
        # note that the above models can accept dropout as an argument

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # for layer in self.layers:
        #     x = layer(x, edge_index)
        #     x = x.relu()
        x = self.gnn(x=x, edge_index=edge_index)
        x = self.norm(x)
        x = x.relu()  # pyg.nn.Models do not apply relu for the output

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin(x)

        return x


class CustomizedHyperGNN(torch.nn.Module):
    def __init__(self, cfg, num_head=4):
        super(CustomizedHyperGNN, self).__init__()

        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train

        model_id = self.model_cfg.model_id
        in_channels = self.dataset_cfg.num_node_features  # since dhg does not support lazy initialization here
        hidden_channels = self.model_cfg.hidden_channels
        num_layer = self.model_cfg.num_layer
        num_class = self.dataset_cfg.num_class
        norm = self.train_cfg.norm

        self.dropout_rate = self.train_cfg.dropout_rate

        if model_id == 'HyperConv':
            self.gnn = HyperConvNN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer,
                                   norm=norm)
            # in_channels ---> hidden_channels
        elif model_id == 'HGNN':
            self.gnn = dhgHGNN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer,
                               norm=norm)
        elif model_id == 'HGNNP':
            self.gnn = dhgHGNNP(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer,
                               norm=norm)
        elif model_id == 'HNHN':
            self.gnn = dhgHNHN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layer,
                               norm=norm)
        else:
            raise NotImplementedError

        self.lin = nn.Linear(in_features=hidden_channels, out_features=num_class)

    def forward(self, x, hyperedge_index, batch):
        # 1. Obtain node embeddings
        x = self.gnn(x=x, hyperedge_index=hyperedge_index)  # no extra normalization/activation is needed
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin(x)

        return x


class GNNPLWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train

        self.model_id = self.model_cfg.model_id
        self.lr = self.train_cfg.lr
        self.num_class = self.dataset_cfg.num_class
        label_smoothing = self.train_cfg.label_smoothing

        if self.model_id in supported_model['GNN']:
            self.model = CustomizedGNN(cfg=cfg)
        else:
            raise NotImplementedError('{} is not implemented'.format(self.model_id))

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = self.model.to(device)
        # self.wct = WeightChangeTracker(model=self.model.lin, model_name=model_id+'_linear')
        if self.num_class == 1:  # regression task
            self.loss_fct = nn.MSELoss()
            if self.dataset_cfg.name in ['esol', 'freesolv', 'lipophilicity']:
                # use RMSE as suggested by https://arxiv.org/pdf/1703.00564.pdf
                self.val_metric = MeanSquaredError(squared=False)
                self.test_metric = MeanSquaredError(squared=False)
            if self.dataset_cfg.name in ['qm9']:
                # use MAE as suggested by https://arxiv.org/pdf/1703.00564.pdf
                self.val_metric = MeanAbsoluteError()
                self.test_metric = MeanAbsoluteError()
        else:
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.val_metric = BinaryAccuracy() if self.num_class == 2 else Accuracy(task="multiclass",
                                                                                    num_classes=self.num_class)
            self.test_metric = BinaryAccuracy() if self.num_class == 2 else Accuracy(task="multiclass",
                                                                                     num_classes=self.num_class)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # self.wct.assert_change()
        x, edge_index, y, batch_vec = batch.x, batch.edge_index, batch.y, batch.batch
        x_hat = self.model(x=x, edge_index=edge_index, batch=batch_vec)

        loss = self.loss_fct(x_hat, y)
        self.log("train_loss", loss)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, edge_index, y, batch_vec = batch.x, batch.edge_index, batch.y, batch.batch
        output = self.model(x=x, edge_index=edge_index, batch=batch_vec)  # (batch_size, num_class)
        if self.num_class == 1:  # regression task
            pred = output.squeeze(-1)
        else:  # classification task
            pred = torch.max(output, dim=1)[1]
        self.val_metric(pred, y)
        return

    def on_validation_epoch_end(self) -> None:
        val_metric_result = self.val_metric.compute()
        self.log("val_metric", val_metric_result)

    def on_test_epoch_start(self) -> None:
        self.test_metric.reset()

    def test_step(self, batch, batch_idx):
        x, edge_index, y, batch_vec = batch.x, batch.edge_index, batch.y, batch.batch
        output = self.model(x=x, edge_index=edge_index, batch=batch_vec)  # (batch_size, num_class)
        if self.num_class == 1:  # regression task
            pred = output.squeeze(-1)
        else:  # classification task
            pred = torch.max(output, dim=1)[1]
        self.test_metric(pred, y)
        return

    def on_test_epoch_end(self):
        test_metric_result = self.test_metric.compute()
        self.log("test_metric", test_metric_result)
        if self.num_class == 1:
            print('test_RMSE/MAE: {}'.format(test_metric_result))
        else:
            print('test_acc: {}'.format(test_metric_result))
        return test_metric_result


class HGNNPLWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train

        self.model_id = self.model_cfg.model_id
        self.lr = self.train_cfg.lr
        self.num_class = self.dataset_cfg.num_class
        label_smoothing = self.train_cfg.label_smoothing

        if self.model_id in supported_model['HGNN']:
            self.model = CustomizedHyperGNN(cfg=cfg)
        else:
            raise NotImplementedError('{} is not implemented'.format(self.model_id))

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = self.model.to(device)
        # self.wct = WeightChangeTracker(model=self.model.lin, model_name='hyperconv'+'_linear')

        if self.num_class == 1:  # regression task
            self.loss_fct = nn.MSELoss()
            if self.dataset_cfg.name in ['esol', 'freesolv', 'lipophilicity']:
                # use RMSE as suggested by https://arxiv.org/pdf/1703.00564.pdf
                self.val_metric = MeanSquaredError(squared=False)
                self.test_metric = MeanSquaredError(squared=False)
            if self.dataset_cfg.name in ['qm9']:
                # use MAE as suggested by https://arxiv.org/pdf/1703.00564.pdf
                self.val_metric = MeanAbsoluteError()
                self.test_metric = MeanAbsoluteError()
        else:
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.val_metric = BinaryAccuracy() if self.num_class == 2 else Accuracy(task="multiclass",
                                                                                    num_classes=self.num_class)
            self.test_metric = BinaryAccuracy() if self.num_class == 2 else Accuracy(task="multiclass",
                                                                                     num_classes=self.num_class)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # self.wct.assert_change()
        x, hyperedge_index, y, batch_vec = batch.x, batch.hyperedge_index, batch.y, batch.batch
        x_hat = self.model(x=x, hyperedge_index=hyperedge_index, batch=batch_vec)

        loss = self.loss_fct(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def validation_step(self, batch, batch_idx):
        x, hyperedge_index, y, batch_vec = batch.x, batch.hyperedge_index, batch.y, batch.batch
        output = self.model(x=x, hyperedge_index=hyperedge_index, batch=batch_vec)
        if self.num_class == 1:  # regression task
            pred = output.squeeze(-1)
        else:  # classification task
            pred = torch.max(output, dim=1)[1]
        self.val_metric(pred, y)
        return

    def on_validation_epoch_end(self) -> None:
        val_metric_result = self.val_metric.compute()
        self.log("val_metric", val_metric_result)

    def on_test_epoch_start(self) -> None:
        self.test_metric.reset()

    def test_step(self, batch, batch_idx):
        x, hyperedge_index, y, batch_vec = batch.x, batch.hyperedge_index, batch.y, batch.batch
        output = self.model(x=x, hyperedge_index=hyperedge_index, batch=batch_vec)
        if self.num_class == 1:  # regression task
            pred = output.squeeze(-1)
        else:  # classification task
            pred = torch.max(output, dim=1)[1]
        self.test_metric(pred, y)
        return

    def on_test_epoch_end(self):
        test_metric_result = self.test_metric.compute()
        self.log("test_metric", test_metric_result)
        if self.num_class == 1:
            print('test_RMSE/MAE: {}'.format(test_metric_result))
        else:
            print('test_acc: {}'.format(test_metric_result))
        return test_metric_result
