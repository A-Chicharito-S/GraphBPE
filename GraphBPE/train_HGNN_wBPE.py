import os
import networkx as nx
from torch_geometric.utils import to_networkx
from trainer import PLTrainerWrapperForGNN
import pickle
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
from omegaconf import DictConfig
import hydra
from supported import supported_dataset
from utils.make_balanced_dataset import split_dataset
from utils.make_hyper_dataset import make_centroid_hypergraph, get_hyperedge_index


class HGNNBPETrainer:
    def __init__(self, cfg):

        self.cfg = cfg

        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train

        self.dataset_name = self.dataset_cfg.name
        self.num_round = self.dataset_cfg.num_round + 1  # +1 for the init preprocessing stage
        self.contract_rings = self.dataset_cfg.contract_rings
        self.contract_cliques = self.dataset_cfg.contract_cliques

        self.model_id = self.model_cfg.model_id
        self.hidden_channels = self.model_cfg.hidden_channels
        self.num_layers = self.model_cfg.num_layer

        self.batch_size = self.train_cfg.batch_size
        self.lr = self.train_cfg.lr
        self.k_fold = self.train_cfg.k_fold

        self.vocab_size_tok = None
        self.vocab_size_origin = None

        if self.dataset_name in supported_dataset.keys():
            self.dataset = supported_dataset[self.dataset_name]
        else:
            raise NotImplementedError('{} is not implemented'.format(self.dataset_name))

        if not self.k_fold:
            self.idx = np.arange(len(self.dataset))
            self.train_index, self.val_index, self.test_index = split_dataset(idx_array=self.idx,
                                                                              dataset_name=self.dataset_name,
                                                                              dataset=self.dataset)
        print('======================train hyper GNN with GraphBPE is called======================')

    def prepare_data(self, round_id):
        file_path = 'tokenized_dataset/{}_contract_rings_{}_cliques_{}'.format(self.dataset_name, self.contract_rings,
                                                                               self.contract_cliques) + '/{}.pickle'.format(
            round_id)

        with open(file_path, 'rb') as f:
            data_infor = pickle.load(f)
            # data_infor = {'dataset': self.tokenized_dataset, 'grammar': self.graph_grammar,
            #                           'vocab': {'ori_size': self.vocab_size_origin, 'ori_tok': self.vocab_origin,
            #                                     'tok_sizes': self.vocab_size_tok}}
            if round_id == self.num_round - 1:
                self.vocab_size_tok = data_infor['vocab']['tok_sizes']
                self.vocab_size_origin = data_infor['vocab']['ori_size']

        tok_dataset = data_infor['dataset']
        processed_data = []

        for data in self.dataset:
            mol_name = data.name
            y = data.y
            node_split = tok_dataset[mol_name]['node_split']

            hyperedge_index = get_hyperedge_index(edge_index=data.edge_index.numpy(),
                                                  node_split=node_split)

            processed_data.append(Data(x=data.x, hyperedge_index=hyperedge_index, y=y))
        num_workers = 0

        if self.k_fold:
            kf = KFold(n_splits=self.k_fold, shuffle=True)
            ori_fold, tok_fold = {}, {}
            for i, (train_index, test_index) in enumerate(kf.split(self.idx)):
                if round_id == 0:
                    hyper_dataset = make_centroid_hypergraph(dataset=self.dataset)
                    ori_train_dataloader = DataLoader(dataset=[hyper_dataset[idx] for idx in train_index],
                                                      batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
                    ori_test_dataloader = DataLoader(dataset=[hyper_dataset[idx] for idx in test_index],
                                                     batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

                    ori_fold[i] = {
                        'train_dict': {'train_dataloader': ori_train_dataloader, 'val_dataloader': ori_test_dataloader},
                        'test_dict': {'test_dataloader': ori_test_dataloader}}

                tok_train_dataloader = DataLoader(dataset=[processed_data[idx] for idx in train_index],
                                                  batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
                tok_test_dataloader = DataLoader(dataset=[processed_data[idx] for idx in test_index],
                                                 batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
                tok_fold[i] = {
                    'train_dict': {'train_dataloader': tok_train_dataloader, 'val_dataloader': tok_test_dataloader},
                    'test_dict': {'test_dataloader': tok_test_dataloader}}
            return ori_fold, tok_fold

        else:
            tok_train_dataloader = DataLoader(dataset=[processed_data[idx] for idx in self.train_index],
                                              batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            tok_val_dataloader = DataLoader(dataset=[processed_data[idx] for idx in self.val_index],
                                            batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
            tok_test_dataloader = DataLoader(dataset=[processed_data[idx] for idx in self.test_index],
                                             batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

            tok_train_dict = {'train_dataloader': tok_train_dataloader, 'val_dataloader': tok_val_dataloader}
            tok_test_dict = {'test_dataloader': tok_test_dataloader}

            if round_id == 0:
                hyper_dataset = make_centroid_hypergraph(dataset=self.dataset)
                ori_train_dataloader = DataLoader(dataset=[hyper_dataset[idx] for idx in self.train_index],
                                                  batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
                ori_val_dataloader = DataLoader(dataset=[hyper_dataset[idx] for idx in self.val_index],
                                                batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
                ori_test_dataloader = DataLoader(dataset=[hyper_dataset[idx] for idx in self.test_index],
                                                 batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

                ori_train_dict = {'train_dataloader': ori_train_dataloader, 'val_dataloader': ori_val_dataloader}
                ori_test_dict = {'test_dataloader': ori_test_dataloader}
                return ori_train_dict, ori_test_dict, tok_train_dict, tok_test_dict

            return tok_train_dict, tok_test_dict

    def get_results(self):
        save_dir = f'results/{self.dataset_name}_contract_rings_{self.contract_rings}_cliques_{self.contract_cliques}/' \
                   f'k_fold{self.k_fold}/{self.model_id}/l{self.num_layers}/lr{self.lr}/b{self.batch_size}/h{self.hidden_channels}'

        if self.dataset_cfg.num_class == 1:  # regression task
            if os.path.exists(save_dir + '/tok_RMSAE.csv') and os.path.exists(save_dir + '/ori_RMSAE.csv'):
                print('found both tok_RMSE.csv and ori_RMSE.csv')
                print('++++++++++++++++++++++{} already run++++++++++++++++++++++'.format(save_dir))
                return
        else:  # classification task
            if os.path.exists(save_dir + '/tok_acc.csv') and os.path.exists(save_dir + '/ori_acc.csv'):
                print('found both tok_acc.csv and ori_acc.csv')
                print('++++++++++++++++++++++{} already run++++++++++++++++++++++'.format(save_dir))
                return

        tok_dataframe, ori_dataframe = {r_id: [] for r_id in range(self.num_round)}, {0: []}
        for round_id in range(self.num_round):
            if self.k_fold:
                if round_id == 0:
                    ori_fold, tok_fold = self.prepare_data(round_id=round_id)
                    for i in range(self.k_fold):
                        ori_fold_i, tok_fold_i = ori_fold[i], tok_fold[i]
                        # not implemented error is handled within the trainer wrapper
                        tok_trainer = PLTrainerWrapperForGNN(type_id='tok',
                                                             round_id='{}_fold_{}'.format(round_id, i),
                                                             cfg=self.cfg)

                        ori_trainer = PLTrainerWrapperForGNN(type_id='ori',
                                                             round_id='{}_fold_{}'.format(round_id, i),
                                                             cfg=self.cfg)

                        ori_trainer.train(train_input_dict=ori_fold_i['train_dict'])
                        tok_trainer.train(train_input_dict=tok_fold_i['train_dict'])

                        ori_acc_i = ori_trainer.test(test_input_dict=ori_fold_i['test_dict'])
                        tok_acc_i = tok_trainer.test(test_input_dict=tok_fold_i['test_dict'])

                        ori_dataframe[round_id].append(ori_acc_i)
                        tok_dataframe[round_id].append(tok_acc_i)

                else:
                    _, tok_fold = self.prepare_data(round_id=round_id)
                    for i in range(self.k_fold):
                        tok_fold_i = tok_fold[i]
                        # not implemented error is handled within the trainer wrapper
                        tok_trainer = PLTrainerWrapperForGNN(type_id='tok',
                                                             round_id='{}_fold_{}'.format(round_id, i),
                                                             cfg=self.cfg)

                        tok_acc_i = tok_trainer.test(test_input_dict=tok_fold_i['test_dict'])

                        tok_dataframe[round_id].append(tok_acc_i)

            else:  # not k_fold
                if round_id == 0:
                    ori_train_dict, ori_test_dict, tok_train_dict, tok_test_dict = self.prepare_data(round_id=round_id)
                    for i in range(5):  # run 5 times
                        # not implemented error is handled within the trainer wrapper
                        tok_trainer = PLTrainerWrapperForGNN(type_id='tok',
                                                             round_id='{}_run_{}'.format(round_id, i),
                                                             cfg=self.cfg)

                        ori_trainer = PLTrainerWrapperForGNN(type_id='ori',
                                                             round_id='{}_run_{}'.format(round_id, i),
                                                             cfg=self.cfg)

                        ori_trainer.train(train_input_dict=ori_train_dict)
                        tok_trainer.train(train_input_dict=tok_train_dict)

                        ori_acc_i = ori_trainer.test(test_input_dict=ori_test_dict)
                        tok_acc_i = tok_trainer.test(test_input_dict=tok_test_dict)

                        ori_dataframe[round_id].append(ori_acc_i)
                        tok_dataframe[round_id].append(tok_acc_i)
                else:
                    tok_train_dict, tok_test_dict = self.prepare_data(round_id=round_id)
                    for i in range(5):  # run 5 times
                        # not implemented error is handled within the trainer wrapper
                        tok_trainer = PLTrainerWrapperForGNN(type_id='tok',
                                                             round_id='{}_run_{}'.format(round_id, i),
                                                             cfg=self.cfg)

                        tok_trainer.train(train_input_dict=tok_train_dict)

                        tok_acc_i = tok_trainer.test(test_input_dict=tok_test_dict)

                        tok_dataframe[round_id].append(tok_acc_i)

        tok_dataframe = pd.DataFrame(data=tok_dataframe)
        ori_dataframe = pd.DataFrame(data=ori_dataframe)

        # save_dir = f'results/{self.dataset_name}_contract_rings_{self.contract_rings}_cliques_{self.contract_cliques}/' \
        #            f'k_fold{self.k_fold}/{self.model_id}/l{self.num_layers}/lr{self.lr}/b{self.batch_size}/h{self.hidden_channels}'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.dataset_cfg.num_class == 1:  # regression task
            tok_dataframe.to_csv(save_dir + '/tok_RMSAE.csv')  # RMSE, MSE, MAE
            ori_dataframe.to_csv(save_dir + '/ori_RMSAE.csv')  # RMSE, MSE, MAE
        else:  # classification task
            tok_dataframe.to_csv(save_dir + '/tok_acc.csv')
            ori_dataframe.to_csv(save_dir + '/ori_acc.csv')

        self.visualization(tok_dict=tok_dataframe, ori_dict=ori_dataframe)

    def visualization(self, tok_dict, ori_dict):
        # matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        Nsteps = self.num_round
        t = np.arange(Nsteps)
        # print(ori_dict)
        # print(tok_dict)
        acc_origin = [ori_dict[0]] * Nsteps  # [num_splits] * num_rounds
        acc_tok = [tok_dict[i] for i in range(Nsteps)]  # [num_splits] * num_rounds

        acc_origin = np.array(acc_origin)  # (num_rounds, num_splits)
        acc_tok = np.array(acc_tok)  # (num_rounds, num_splits)

        # populations over time
        mu1 = acc_origin.mean(axis=1)
        sigma1 = acc_origin.std(axis=1)
        mu2 = acc_tok.mean(axis=1)
        sigma2 = acc_tok.std(axis=1)

        vis_dir = f'visualization/{self.dataset_name}_contract_rings_{self.contract_rings}_cliques_{self.contract_cliques}/k_fold{self.k_fold}/{self.model_id}'

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        # plot it!

        fig, ax = plt.subplots(1)
        ax.plot(t, mu1, lw=2, label='original', color='blue')
        ax.plot(t, mu2, lw=2, label='tokenized', color='red')
        ax.fill_between(t, mu1 + sigma1, mu1 - sigma1, facecolor='blue', alpha=0.5)
        ax.fill_between(t, mu2 + sigma2, mu2 - sigma2, facecolor='red', alpha=0.5)
        ax.set_title(r'[{}-acc] $\mu$ and $\pm \sigma$ interval'.format(self.model_id))
        ax.legend(loc='upper left')
        ax.set_xlabel('num tokenization steps')
        if self.dataset_cfg.num_class == 1:
            if self.dataset_name in ['esol', 'freesolv', 'lipophilicity']:
                ax.set_ylabel('RMSE')
            else:  # qm9
                ax.set_ylabel('MAE')
        else:
            ax.set_ylabel('acc')
        major_ticks = np.arange(0, Nsteps, 10)
        ax.set_xticks(major_ticks)
        ax.grid()
        pic_name = f'l{self.num_layers}_lr{self.lr}_b{self.batch_size}_h{self.hidden_channels}_acc.png'
        plt.savefig(vis_dir + '/' + pic_name, dpi=300)

        fig, ax = plt.subplots(1)
        o_vocab = np.array([self.vocab_size_origin] * Nsteps)
        t_vocab = np.array(self.vocab_size_tok)

        ax.plot(t, o_vocab, lw=2, label='origin-vocab', color='blue')
        ax.plot(t, t_vocab, lw=2, label='subgraph-vocab', color='red')

        ax.set_title(r'vocab size')
        ax.legend(loc='upper left')
        ax.set_xlabel('num tokenization steps')
        ax.set_ylabel('vocab size')
        major_ticks = np.arange(0, Nsteps, 10)
        ax.set_xticks(major_ticks)
        ax.grid()
        if self.contract_cliques and self.contract_rings:
            pic_name = 'contract_rings_cliques_vocab_size.png'
        elif self.contract_rings and not self.contract_cliques:
            pic_name = 'contract_rings_vocab_size.png'
        elif self.contract_cliques and not self.contract_rings:
            pic_name = 'contract_cliques_vocab_size.png'
        else:
            pic_name = 'keep_rings_cliques_vocab_size.png'
        plt.savefig(vis_dir + '/' + pic_name, dpi=300)


@hydra.main(version_base='1.3', config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    hgnn_trainer = HGNNBPETrainer(cfg=cfg)
    hgnn_trainer.get_results()


if __name__ == '__main__':
    main()
