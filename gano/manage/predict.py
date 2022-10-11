import argparse
import logging
import os
from abc import ABC, abstractmethod

# This environment variable needs to be initialized before importing Pytorch Lightning
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import pytorch_lightning as pl

from gano.manage.experiments import GMExperiment, GMExperimentLMMixin

class GMPredictor(GMExperiment, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parser = argparse.ArgumentParser()
        self.add_args()

        self.args = self.parser.parse_args()
        self.experiment_name = self.args.experiment

        self.init_dirs()
        self.init_dataset()
    
    def add_args(self, defaults: dict = None) -> None:
        add = lambda *args, **kwargs: self.parser.add_argument(*args, **kwargs)

        add('experiment', type=str, help='Experiment name')
        add('--batch', type=int, default=32, help='Batch size')
        add('--cache', action='store_true', help='Use cache for preprocessing')
        add('--gpus', type=str, default=None, help='e.g., "0,"')
        add('--no-record', action='store_true', help="Do not save model's outputs")
        add('--workers', type=int, default=12, help='Number of workers for data loader')
    
    def build_dataset_params(self) -> dict:
        return {
            'batch_size': self.args.batch,
            'cache_dir': self.cache_dir,
            'data_dir': self.data_dir,
            'experiment_name': self.args.experiment,
            'model_dir': self.model_dir,
            'num_workers': self.args.workers,
            'stage': 'predict',
            'use_cache': self.args.cache}
    
    def build_model_params(self) -> dict:
        return {
            'experiment_name': self.experiment_name,
            'log_dir': self.log_dir,
            'model_dir': self.model_dir,
            'output_dir': self.output_dir,
            'no_record': self.args.no_record,
            'stage': 'predict',
            'test_dir': self.test_dir}
    
    def init_dataset(self):
        self.data_params = self.build_dataset_params()
        self.dataset = self.dataset_cls(self.args.model)(**self.data_params)
    
    @abstractmethod
    def predict(self) -> None:
        raise NotImplementedError


class GMPredictorLMMixin(GMPredictor, GMExperimentLMMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model = self.args.experiment.split('.')[1]
        self.model_name = self.MODELS[model]
        self.model_params = self.build_model_params()

        self.init_lm_dirs()

        self.data_params = self.build_dataset_params()
        self.dataset = self.dataset_cls(model)(**self.data_params)

        if self.args.debug_dataset:
            self.dataset.setup('predict')
            exit()
        
        self.model = self.model_cls(model)(dataset=self.dataset, **self.model_params)

        logging.info(f'Experiment: {self.experiment_name}')
    
    def add_args(self, defaults: dict = None) -> None:
        super().add_args(defaults)

        add = lambda *args, **kwargs: self.parser.add_argument(*args, **kwargs)
        add('--debug-dataset', action='store_true', help='Debug the dataset and do not load the model')

    def build_dataset_params(self) -> dict:
        return {
            'model_name': self.model_name,
            **super().build_dataset_params()}
    
    def build_model_params(self) -> dict:
       return {
            'model_name': self.model_name,
            **super().build_model_params()}
    
    def init_dataset(self):
        pass

    def predict(self) -> None:
        self.dataset.setup('predict')

        self.trainer = pl.Trainer(
            gpus=self.args.gpus,
            default_root_dir=self.pl_dir)
        
        outputs = self.trainer.predict(
            self.model, 
            dataloaders=self.dataset.predict_dataloader)
        
        self.model.predict_epoch_end(outputs)
