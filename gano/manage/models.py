import csv
import json
import logging
import pickle
from abc import ABC, abstractmethod
from os.path import join
from typing import Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import wandb
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from gano.manage.datasets import GMDataset, GMDatasetLMMixin
from gano.manage.utils import update_args


class GMLightning(pl.LightningModule, ABC):
    def __init__(
        self,
        dataset: GMDataset = None,
        experiment_name: str = None,
        full_train: bool = False,
        fork: str = None,
        gpus: str = None,
        log_dir: str = None,
        lr: float = None,
        model_dir: str = None,
        monitor: str = None,
        monitor_rule: str = None,
        no_checkpoint: bool = False,
        no_record: bool = False,
        output_dir: str = None,
        stage: str = 'fit',
        test_dir: str = None,
        warmup: float = .1,
        wb: bool = None):

        super().__init__()

        self.dataset = dataset
        self.experiment_name = experiment_name
        self.full_train = full_train
        self.fork = fork
        self.gpus = gpus
        self.monitor = monitor
        self.monitor_rule = monitor_rule
        self.no_checkpoint = no_checkpoint
        self.no_record = no_record
        self.stage = stage
        self.wb = wb

        self.log_dir = log_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.test_dir = test_dir

        self.config_cls = None
        self.encoder_cls = None
        self.model_cls = None
        self.model = None

        self.is_sanity_check = True
        self.log_outputs_this_epoch = False
        self.results = {'train': [], 'val': [], 'test': [], 'predict': []}
        self.outputs = {'train': [], 'val': [], 'test': [], 'predict': []}
        self.losses = {'train': [], 'val': [], 'test': [], 'predict': []}

        self.hparams['learning_rate'] = lr
        self.hparams['warmup'] = warmup

        if monitor_rule == 'min':
            self.best_metric_value = float('inf')
        
        else:
            self.best_metric_value = -float('inf')
    
    def configure_optimizers(self):
        num_training_steps = self._get_total_train_steps()
        num_warmup = int(num_training_steps * self.hparams.warmup)

        params = self._get_model_params()

        optimizer = AdamW(params, lr=self.hparams.learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup,
            num_training_steps)

        return [optimizer], [scheduler]
    
    def test_step(self, batch: tuple, batch_idx: int):
        return self.train_test_step(batch, batch_idx, 'test')
    
    def test_epoch_end(self, test_step_outputs: list):
        self.outputs['test'] = test_step_outputs
        self._epoch_end('test')
    
    def training_step(self, batch: tuple, batch_idx: int):
        return self.train_test_step(batch, batch_idx, 'train')
    
    def training_epoch_end(self, training_step_outputs: list):
        self.outputs['train'] = training_step_outputs
        self._epoch_end('train')
    
    def validation_step(self, batch: tuple, batch_idx: int):
        return self.train_test_step(batch, batch_idx, 'val')
    
    def validation_epoch_end(self, validation_step_outputs: list):
        if self.is_sanity_check:
            self.is_sanity_check = False
            return
        
        self.outputs['val'] = validation_step_outputs

    def clean_outputs(self) -> None:
        self.outputs = {'train': [], 'val': [], 'test': []}
    
    def combine_outputs(self, outputs: list, key: str = 'preds') -> dict:
        output = {}

        for batch in outputs:
            output = {**output, **batch[key]}
        
        return output
    
    def log_losses(self, split: str) -> None:
        for split in self._get_splits(split):
            keys = [k for k in self.outputs[split][-1].keys() if k.startswith('loss')]
            losses = {}

            for key in keys:
                loss = [o[key] for o in self.outputs[split] if o[key] is not None]
                losses[key] = sum(loss) / len(loss) if len(loss) else 0
            
            self.losses[split].append(losses)
        
            if self.wb:
                wandb.log(
                    {f'{split}_{k}': v for k, v in losses.items()},
                    step=self.current_epoch)
            
            else:
                output = f'Loss [{split}]:\n'
                
                for key, value in losses.items():
                    output += '.. %s: %.4f\n' % (key, value)
                
                logging.info(output)
        
        self.save_loss(split)
    
    def log_make_decision(self, split: str) -> None:
        if split in ('test', 'predict') or self.monitor is None or self.monitor_rule is None:
            self.log_outputs_this_epoch = True
            return

        self.log_outputs_this_epoch = False
        metric = self.results['val'][-1][self.monitor]

        if (self.monitor_rule == 'max' and self.best_metric_value < metric) or \
            (self.monitor_rule == 'min' and self.best_metric_value > metric):
            self.log_outputs_this_epoch = True
            self.best_metric_value = metric
    
    def log_outputs(self, split: str) -> None:
        if self.log_outputs_this_epoch:
            output_path = join(self.output_dir, f'info.json')

            for split in self._get_splits(split):
                if 'preds' in self.outputs[split]:
                    output = self.outputs[split]['preds']
                    self.save_preds(output, split)
                
                if 'records' in self.outputs[split]:
                    record = self.outputs[split]['records']
                    self.save_record(record, split)
                
                if 'results' in self.outputs[split]:
                    record = self.outputs[split]['results']
                    self.save_evals(record, split)

            if split == 'train':
                with open(output_path, 'w') as writer:
                    json.dump({'fit_best_epoch': self.current_epoch}, writer, indent=2)
    
    def log_results(self, split: str) -> None:
        for split in self._get_splits(split):
            if self.wb:
                wandb.log(
                    {f'{split}_{k}': v for k, v in self.results[split][-1].items()},
                    step=self.current_epoch)

            else:
                output = f'Evaluation [{split}]\n'

                for key, value in self.results[split][-1].items():
                    if isinstance(value, float):
                        value = '%.4f' % value
                    
                    output += f'.. {key}: {value}\n'

                logging.info(output)
        
        self.save_results(split)

    def predict_epoch_end(self, predict_step_outputs: list):
        self.outputs['predict'] = predict_step_outputs
        self._epoch_end('predict')

    def save_loss(self, split: str) -> None:
        if self.output_dir is not None:
            for split in self._get_splits(split):
                if len(self.losses[split]):
                    output_path = join(self.output_dir, f'loss-{split}.csv')

                    with open(output_path, 'w') as writer:
                        cw = csv.DictWriter(
                            writer, 
                            fieldnames=list(self.losses[split][-1].keys()))
                        
                        cw.writeheader()

                        for loss in self.losses[split]:
                            cw.writerow({k: float(v) for k, v in loss.items()})

    def save_preds(self, preds: dict, split: str) -> None:
        if self.output_dir is not None:
            output_path = join(self.output_dir, f'{split}.json')

            with open(output_path, 'w') as writer:
                json.dump(preds, writer, indent=2)
    
    def save_record(self, output: dict, split: str) -> None:
        if not self.no_record and self.output_dir is not None:
            output_path = join(self.output_dir, f'{split}.pkl')

            with open(output_path, 'wb') as writer:
                pickle.dump(output, writer)
    
    def save_evals(self, preds: dict, split: str) -> None:
        if self.output_dir is not None:
            output_path = join(self.output_dir, f'result-{split}.json')

            with open(output_path, 'w') as writer:
                json.dump(preds, writer, indent=2)
    
    def save_results(self, split: str) -> None:
        if self.output_dir is not None:
            for split in self._get_splits(split):
                if len(self.results[split]):
                    output_path = join(self.output_dir, f'result-{split}.csv')

                    with open(output_path, 'w', newline='') as writer:
                        cw = csv.DictWriter(
                            writer, 
                            fieldnames=list(self.results[split][-1].keys()))

                        cw.writeheader()

                        for result in self.results[split]:
                            cw.writerow(result)

    def update_args(self, default: dict, override: dict) -> dict:
        return update_args(default, override)
    
    def _epoch_end(self, split: str) -> None:
        if split != 'predict':
            self.log_losses(split)

        self.epoch_end(split)
        self.log_make_decision(split)
        self.log_outputs(split)

        if split != 'predict':
            self.log_results(split)
            
        self.clean_outputs()
    
    def _get_total_train_steps(self):
        data = self.dataset.train_dataloader()
        return int(len(data) / self.trainer.accumulate_grad_batches) * self.trainer.max_epochs
    
    def _get_model_params(self):
        params_with_decay = []
        params_without_decay = []
        n_decay = n_no_decay = 0

        for name, params in self.model.named_parameters():
            if 'bias' not in name:
                params_with_decay.append(params)
                n_decay += len(params)
            else:
                params_without_decay.append(params)
                n_no_decay += len(params)
        
        logging.info(f'Parameters with decay: {n_decay}')
        logging.info(f'Parameters without decay: {n_no_decay}')
        
        return [
            {'params': params_with_decay, 'weight_decay': 0.01},
            {'params': params_without_decay, 'weight_decay': 0.0}]
    
    def _get_splits(self, split: str):
        return ('train', 'val') if split in ('train', 'val') else (split,)

    @abstractmethod
    def epoch_end(self, split: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def train_test_step(self, batch: tuple, batch_idx: int) -> dict:
        raise NotImplementedError


class GMLightningLMMixin(GMLightning):
    def __init__(
        self, 
        model_name: str = None, 
        dataset: GMDatasetLMMixin = None,
        tokenizer: PreTrainedTokenizer = None,
        *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.config: PretrainedConfig = ...
        self.encoder: PreTrainedModel = ...
        self.model: nn.Module = ...

        self.config_cls: Type[PretrainedConfig] = ...
        self.encoder_cls: Type[PreTrainedModel] = ...

        self.dataset = dataset
        self.model_name = model_name
        self.tokenizer = tokenizer

        self.init_model()

    def init_model(self, model_params: dict = None) -> None:
        self.init_model_cls()

        if self.fork is not None:
            self.config = self.config_cls.from_pretrained(join(self.fork, 'config'))
            self.encoder = self.encoder_cls(config=self.config)
            logging.info(f"Loaded model from: {self.fork}")
    
        elif self.stage not in ('test', 'predict'):
            self.config = self.config_cls.from_pretrained(self.model_name)
            self.config = self.init_model_config(self.config)
            self.encoder = self.encoder_cls.from_pretrained(self.model_name, config=self.config)
            logging.info(f'Loaded pre-trained model: {self.model_name}')
            
        else:
            self.config = self.config_cls.from_pretrained(join(self.model_dir, 'config'))
            self.encoder = self.encoder_cls.from_config(self.config)
            logging.info(f"Loaded model from: {self.model_dir}")

        if model_params is None: 
            model_params = {}

        self.model: nn.Module = self.model_cls(self.encoder, self.config, **model_params)
        self.encoder.resize_token_embeddings(len(self.dataset.tokenizer))

        if self.stage in ('test', 'predict'):
            device = 'cpu' if self.gpus is None else f"cuda:{self.gpus.strip(',')}"
            self.model.load_state_dict(torch.load(join(self.model_dir, f'model.pt'), map_location=device))
            logging.info(f"Loaded model from: {join(self.model_dir, f'model.pt')}")
    
    def log_outputs(self, split: str) -> None:
        super().log_outputs(split)

        if split not in ('test', 'predict') and self.log_outputs_this_epoch:
            self.save_model()
    
    def save_model(self) -> None:
        if not self.no_checkpoint and self.model_dir is not None:
            torch.save(self.model.state_dict(), join(self.model_dir, f'model.pt'))
            self.dataset.tokenizer.save_pretrained(join(self.model_dir, 'tokenizer'))
            self.config.save_pretrained(join(self.model_dir, 'config'))

    def init_model_cls(self) -> None:
        self.config_cls = AutoConfig
        self.encoder_cls = AutoModel

    def init_model_config(self, config: PretrainedConfig) -> PretrainedConfig:
        return config
