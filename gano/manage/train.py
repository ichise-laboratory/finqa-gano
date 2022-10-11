import argparse
import logging
import os
from abc import ABC, abstractmethod
from os.path import exists, join

# This environment variable needs to be initialized before importing Pytorch Lightning
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import pytorch_lightning as pl
import wandb

from gano.manage.experiments import GMExperiment, GMExperimentLMMixin
from gano.manage.utils import LightningProgressBar, update_args


class GMTrainer(GMExperiment, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parser = argparse.ArgumentParser()
        self.add_args()

        self.iter: int = None
        self.flags: str = ...

        self.args = self.parser.parse_args()
        
        self.init_flags()
        self.experiment_name = self.create_experiment()

        self.init_dirs()
        self.init_wb()
        self.init_dataset()

    def add_args(self, defaults: dict = None) -> None:
        get = lambda key, default: defaults[key] if defaults is not None and key in defaults else default
        add = lambda *args, **kwargs: self.parser.add_argument(*args, **kwargs)

        add('--acc-grad', type=int, default=1, help='Accumulate gradients')
        add('--batch', type=int, default=32, help='Batch size')
        add('--cache', action='store_true', help='Use cache for preprocessing')
        add('--clip-grad', type=float, default=None, help='Clip gradients')
        add('--fork', type=str, default=None, help='Create and train a copy of a fine-tuned model')
        add('--debug-dataset-ratio', type=float, default=None, help='Process a portion of the training/validation data')
        add('--debug-ratio', type=float, default=None, help='Use a portion of the training/validation data')
        add('--epochs', type=int, default=5, help='Training epochs')
        add('--flags', type=str, default='', help='Special experiment flags')
        add('--full-train', action='store_true', help='Train on full dataset')
        add('--gpus', type=str, default=None, help='e.g., "0,"')
        add('--iter', action='store_true', help='Repeated traning')
        add('--iter-start', type=int, default=1, help='Starting iteration index')
        add('--lr', type=float, default=get('lr', 1e-3), help='Learning rate')
        add('--monitor', type=str, default=get('monitor', 'f1'), help='Monitor a metric for checkpointing')
        add('--monitor-rule', type=str, default=get('monitor_rule', 'max'), help='A condition to save checkpoint')
        add('--no-checkpoint', action='store_true', help='Do not save best checkpoint')
        add('--no-record', action='store_true', help="Do not save model's outputs")
        add('--pipeline', type=str, default=self.pipeline.name, help='Extra parameter for W&B')
        add('--template', type=str, default=get('template', 'f.v.e.l.i'), help='Template for experiment name')
        add('--val-index', type=int, default=0, help='Cross-validation partition index')
        add('--val-folds', type=int, default=None, help='Number of cross-validation partitions')
        add('--warmup', type=float, default=get('warmup', .1), help='Warmup ratio')
        add('--wb', action='store_true', help='Enable Weights and Biases')
        add('--workers', type=int, default=12, help='Number of workers for data loader')

    def build_dataset_params(self) -> dict:
        return {
            'batch_size': self.args.batch,
            'cache_dir': self.cache_dir,
            'data_dir': self.data_dir,
            'debug_dataset_ratio': self.args.debug_dataset_ratio,
            'debug_ratio': self.args.debug_ratio,
            'experiment_name': self.experiment_name,
            'fork': self.args.fork,
            'full_train': self.args.full_train,
            'model_dir': self.model_dir,
            'num_workers': self.args.workers,
            'output_dir': self.output_dir,
            'stage': 'fit',
            'use_cache': self.args.cache,
            'val_index': self.args.val_index}
    
    def build_model_params(self) -> dict:
        return {
            'experiment_name': self.experiment_name,
            'full_train': self.args.full_train,
            'fork': self.args.fork,
            'log_dir': self.log_dir,
            'lr': self.args.lr,
            'model_dir': self.model_dir,
            'monitor': self.args.monitor,
            'monitor_rule': self.args.monitor_rule,
            'no_checkpoint': self.args.no_checkpoint,
            'no_record': self.args.no_record,
            'output_dir': self.output_dir,
            'stage': 'fit',
            'warmup': self.args.warmup,
            'wb': self.args.wb}
    
    def init_dataset(self) -> None:
        self.data_params = self.build_dataset_params()
        self.dataset = self.dataset_cls(self.args.model)(**self.data_params)

        logging.info(f'Dataset arguments: {self.data_params}')

    def init_flags(self) -> None:
        self.flags = self.args.flags

        if self.args.full_train:
            self.flags = self.flags + '-f' if len(self.flags) else 'f'
    
    def init_wb(self) -> None:
        if self.args.wb:
            project = self.PROJECT.replace('/', '-')
            task = self.TASK.replace('/', '-') if self.TASK is not None else None

            if task is not None:
                name = f'{self.DOMAIN}-{project}-{task}'
            else:
                name = f'{self.DOMAIN}-{project}'

            wandb.init(
                project=name,
                name=self.experiment_name,
                dir=self.wb_dir,
                config=self.args.__dict__)

    def update_args(self, default: dict, override: dict) -> dict:
        return update_args(default, override)
    
    @abstractmethod
    def create_experiment(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError


class GMTrainerLMMixin(GMTrainer, GMExperimentLMMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = self.MODELS[self.args.model]
        self.model_params = self.build_model_params()

        self.init_lm_dirs()

        self.data_params = self.build_dataset_params()
        self.dataset = self.dataset_cls(self.args.model)(**self.data_params)

        if self.args.debug_dataset:
            self.dataset.setup('fit')
            exit()
        
        self.model = self.model_cls(self.args.model)(dataset=self.dataset, **self.model_params)

        logging.info(f'Experiment: {self.experiment_name}')

    def add_args(self, defaults: dict = None) -> None:
        defaults = self.update_args(defaults, {'template': 'f.v.e.l.i'})
        super().add_args(defaults)

        add = lambda *args, **kwargs: self.parser.add_argument(*args, **kwargs)
        add('model', type=str)
        add('--debug-dataset', action='store_true', help='Debug the dataset and do not initialize the model')

    def build_dataset_params(self) -> dict:
        return {
            'model_name': self.model_name,
            **super().build_dataset_params()}
    
    def build_model_params(self) -> dict:
       return {
            'model_name': self.model_name,
            **super().build_model_params()}
    
    def create_experiment(self) -> str:
        encode = lambda s: s.replace('.', '$dot$') \
            .replace('*', '$star$')
        decode = lambda s: s.replace('$dot$', '.') \
            .replace('$star$', '*')

        lr = ('%.0e' % self.args.lr).replace('-0', '-')
        batch = self.args.batch
        epochs = str(self.args.epochs)
        flags = encode(self.flags)
        model = encode(self.args.model)
        pipeline = encode(self.pipeline.symbol)
        template = f'.p.m.{self.args.template}'
        warmup = str(self.args.warmup)

        if self.args.val_folds is not None:
            val = f'{self.args.val_index}-{self.args.val_folds}'

        else:
            val = f'{self.args.val_index}'

        name = template \
            .replace('.b', f'*b{batch}') \
            .replace('.e', f'*e{epochs}') \
            .replace('.f', f'*$flags$') \
            .replace('.l', f'*{lr}') \
            .replace('.m', f'*{model}') \
            .replace('.p', f'*{pipeline}') \
            .replace('.v', f'*v{val}') \
            .replace('.w', f'*w{warmup}') \
            .replace('*', '.') \
            .replace('..', '.') \
            .strip('.')
        
        if not self.args.iter:
            return decode(name.replace('.i', '').replace('$flags$', flags)).replace('..', '.')
        
        else:
            self.iter = self.args.iter_start
            
            while True:
                task = self.TASK or ''
                iter_name = decode(name.replace('.i', f'.i-{self.iter}') \
                    .replace('$flags$', flags)).replace('..', '.')
                path = join(self.OUTPUT_DIR, self.PROJECT, task, self.pipeline.name, iter_name)

                if exists(path):
                    self.iter += 1
                
                else:
                    return iter_name
    
    def init_dataset(self):
        pass
    
    def train(self) -> None:
        progress_bar = LightningProgressBar()

        self.trainer = pl.Trainer(
            deterministic=False, 
            gpus=self.args.gpus, 
            max_epochs=self.args.epochs,
            default_root_dir=self.pl_dir,
            enable_checkpointing=False,
            accumulate_grad_batches=self.args.acc_grad,
            gradient_clip_val=self.args.clip_grad,
            callbacks=[progress_bar])
        
        self.trainer.fit(self.model, datamodule=self.dataset)
