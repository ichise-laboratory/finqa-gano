import argparse
import logging
import pathlib
from abc import ABC, abstractmethod
from collections import namedtuple
from os.path import join
from typing import Dict

from gano.manage.datasets import GMDataset, GMDatasetLMMixin
from gano.manage.models import GMLightning, GMLightningLMMixin

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class GMExperiment(ABC):
    DOMAIN: str = ...
    PROJECT: str = ...
    TASK: str = None
    PIPELINE: str = ...

    CACHE_DIR: str = 'cache'
    DATA_DIR: str = 'data'
    LOG_DIR: str = 'logs'
    MODEL_DIR: str = 'models'
    OUTPUT_DIR: str = 'outputs'
    TEST_DIR: str = 'tests'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.args: argparse.Namespace = ...
        self.experiment_name: str = ...

    def init_dirs(self) -> None:
        task = self.TASK or ''
        self.cache_dir = join(self.CACHE_DIR, self.DOMAIN, self.PROJECT, task, self.pipeline.name)
        self.data_dir = join(self.DATA_DIR, self.DOMAIN, self.PROJECT, task)
        self.log_dir = join(self.LOG_DIR, self.DOMAIN, self.PROJECT, task, self.pipeline.name, self.experiment_name)
        self.model_dir = join(self.MODEL_DIR, self.DOMAIN, self.PROJECT, task, self.pipeline.name, self.experiment_name)
        self.output_dir = join(self.OUTPUT_DIR, self.DOMAIN, self.PROJECT, task, self.pipeline.name, self.experiment_name)
        self.pl_dir = join(self.LOG_DIR, self.DOMAIN, self.PROJECT, task, self.pipeline.name, self.experiment_name, 'pl')
        self.test_dir = join(self.TEST_DIR, self.DOMAIN, self.PROJECT, task, self.pipeline.name, self.experiment_name)
        self.wb_dir = join(self.LOG_DIR, self.DOMAIN, self.PROJECT, task, self.pipeline.name, self.experiment_name, 'wb')

        for path in (self.log_dir, self.output_dir, self.pl_dir):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        
        if self.args.cache:
            pathlib.Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        if hasattr(self.args, 'wb') and self.args.wb:
            pathlib.Path(self.wb_dir).mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.args, 'no_checkpoint') and not self.args.no_checkpoint:
            pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def pipeline(self) -> namedtuple:
        Pipeline = namedtuple('Pipeline', ['symbol', 'name'])
        return Pipeline(*self.PIPELINE.split(':'))
    
    @abstractmethod
    def dataset_cls(self, model: str) -> GMDataset:
        raise NotImplementedError
    
    @abstractmethod
    def model_cls(self, model: str) -> GMLightning:
        raise NotImplementedError


class GMExperimentLMMixin(GMExperiment):
    MODELS: Dict[str, str] = ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name: str = ...
    
    def init_lm_dirs(self) -> None:
        task = self.TASK or ''
        self.cache_dir = join(self.CACHE_DIR, self.PROJECT, task, self.pipeline.name, self.model_name)

        if self.args.cache:
            pathlib.Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def invalid_model(self, model: str) -> None:
        raise RuntimeError(f'Invalid model "{model}". \
            Valid choices are {str(list(self.MODELS.keys()))}.')

    @abstractmethod
    def dataset_cls(self, model: str) -> GMDatasetLMMixin:
        raise NotImplementedError
    
    @abstractmethod
    def model_cls(self, model: str) -> GMLightningLMMixin:
        raise NotImplementedError
