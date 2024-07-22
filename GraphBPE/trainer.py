import os
import shutil

import pytorch_lightning as pl
import torch
from model import GNNPLWrapper, HGNNPLWrapper
from pytorch_lightning.callbacks import ModelCheckpoint
from supported import supported_model



class PLTrainerWrapperForGNN:
    def __init__(self, type_id, round_id, cfg):

        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train

        self.dataset_name = self.dataset_cfg.name
        self.contract_rings = self.dataset_cfg.contract_rings
        self.contract_cliques = self.dataset_cfg.contract_cliques

        self.devices = self.train_cfg.num_device
        self.max_epochs = self.train_cfg.max_epochs
        self.type_id = type_id  # 'ori' / 'tok'
        self.round_id = round_id
        self.model_id = self.model_cfg.model_id

        self.hidden_channels = self.model_cfg.hidden_channels
        self.num_layers = self.model_cfg.num_layer
        self.lr = self.train_cfg.lr
        self.k_fold = self.train_cfg.k_fold

        if self.model_id in supported_model['GNN']:
            self.model = GNNPLWrapper(cfg=cfg)
        elif self.model_id in supported_model['HGNN']:
            self.model = HGNNPLWrapper(cfg=cfg)
        else:
            raise NotImplementedError

        # declare the trainer
        save_dir = f'checkpoints/{self.dataset_name}_contract_rings_{self.contract_rings}_cliques_{self.contract_cliques}/k_fold{self.k_fold}_m{self.model_id}/{self.type_id}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ckpt_dir_name = 'r{}_l{}_h{}_lr{}'.format(self.round_id, self.num_layers, self.hidden_channels, self.lr)
        if self.dataset_cfg.num_class == 1:
            save_mode = 'min'  # minimize RMSE/MSE/MAE
        else:
            save_mode = 'max'  # maximize acc

        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir + '/{}'.format(ckpt_dir_name),
            filename='{epoch}',
            monitor='val_metric',
            save_top_k=1,
            mode=save_mode,
            every_n_epochs=1)

        self.trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=self.devices,
            max_epochs=self.max_epochs,
            strategy="auto",
            enable_progress_bar=False,  # True,
            callbacks=[checkpoint_callback],
            logger=[],
            # gradient_clip_val=1
        )

    def train(self, train_input_dict):
        # train_input_dict: {'train_dataloader': xxx, 'val_dataloader': xxx}

        train_dataloader, val_dataloader = train_input_dict['train_dataloader'], train_input_dict['val_dataloader']
        self.trainer.fit(model=self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    def test(self, test_input_dict):
        # test_input_dict: {'test_dataloader': xxx}

        # results = {}

        save_dir = f'checkpoints/{self.dataset_name}_contract_rings_{self.contract_rings}_cliques_{self.contract_cliques}/k_fold{self.k_fold}_m{self.model_id}/{self.type_id}'
        if not self.k_fold:  # load the best ckpt if we are not doing cross validation
            ckpt_dir_name = 'r{}_l{}_h{}_lr{}'.format(self.round_id, self.num_layers, self.hidden_channels, self.lr)

            best_ckpt = [f for f in os.listdir(save_dir + '/{}'.format(ckpt_dir_name))]
            assert len(best_ckpt) == 1
            # print('=====================best_ckpt{}========================='.format(best_ckpt))
            best_ckpt = save_dir + '/{}/'.format(ckpt_dir_name) + best_ckpt[0]
            # results['best_ckpt'] = best_ckpt

            checkpoint = torch.load(
                best_ckpt,
                map_location=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
            )
            self.model.load_state_dict(checkpoint['state_dict'])

        test_dataloader = test_input_dict['test_dataloader']
        test_metric_results = self.trainer.test(model=self.model, dataloaders=test_dataloader)
        test_metric_results = test_metric_results[0]['test_metric']

        # remove the previously saved ckpt
        if not self.k_fold:
            ckpt_dir_name = 'r{}_l{}_h{}_lr{}'.format(self.round_id, self.num_layers, self.hidden_channels, self.lr)
            shutil.rmtree(save_dir + '/{}'.format(ckpt_dir_name))
        else:
            shutil.rmtree(save_dir)

        return test_metric_results
