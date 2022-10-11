import json
import logging
from abc import ABC, abstractmethod
from os.path import exists, join
from pathlib import Path
from typing import Tuple, Type

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


class GMDataset(pl.LightningDataModule, ABC):
    def __init__(
        self, 
        batch_size: int = 32,
        cache_dir: str = None,
        data_dir: str = None,
        debug_dataset_ratio: float = None,
        debug_ratio: float = None,
        experiment_name: str = None,
        fork: str = None,
        full_train: bool = False,
        model_dir: str = None,
        model_name: str = None, 
        num_workers: int = 12,
        output_dir: str = None,
        stage: str = 'fit',
        test_dir: str = None,
        val_index: int = 0,
        use_cache: bool = False,
        *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.debug_dataset_ratio = debug_dataset_ratio
        self.debug_ratio = debug_ratio
        self.experiment_name = experiment_name
        self.fork = fork
        self.full_train = full_train
        self.model_dir = model_dir
        self.model_name = model_name
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.stage = stage
        self.test_dir = test_dir
        self.use_cache = use_cache
        self.val_index = val_index

        self.samples = {'train': None, 'val': None, 'test': None, 'predict': None}
        self.tensors = {'train': None, 'val': None, 'test': None, 'predict': None}
        self.ids_map = {'train': None, 'val': None, 'test': None, 'predict': None}
    
    def preprocess(self, split: str, samples: dict) -> dict:
        return samples
    
    def setup(self, stage=None):
        if self.stage in (None, 'fit'):
            self._setup(('train', 'val'))
        
        if self.stage in (None, 'test'):
            self._setup(('test',))
        
        if self.stage in (None, 'predict'):
            self._setup(('predict',))
    
    def test_dataloader(self):
        return DataLoader(
            self.tensors['test'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers)
    
    def train_dataloader(self):
        return DataLoader(
            self.tensors['train'], 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.tensors['val'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers)
    
    def _setup(self, splits: list) -> None:
        for split in splits:
            cache_path = join(self.cache_dir, f'{split}.pt')

            self.samples[split] = self.load(split)
            self.samples[split] = self.preprocess(split, self.samples[split])

            if self.use_cache and exists(cache_path):
                self.tensors[split] = torch.load(cache_path)

            else:
                if self.debug_dataset_ratio is not None:
                    logging.warning('Processing only %.2f percent of the data.' % (self.debug_dataset_ratio * 100))
                    reduced_size = int(len(self.samples[split]) * self.debug_dataset_ratio)
                    self.samples[split] = self.samples[split][:reduced_size]

                self.tensors[split], self.ids_map[split] = self.process(split, self.samples[split])

                if self.use_cache:
                    Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(self.tensors[split], cache_path)
            
            self._setup_aux(split)
        
        if self.debug_ratio is not None:
            limit = int(len(self.tensors['train']) * self.debug_ratio)
            self.tensors['train'] = TensorDataset(*[t[:limit] for t in self.tensors['train'].tensors])
    
    def _setup_aux(self, split: str) -> None:
        aux_path = join(self.cache_dir, f'{split}.json')

        if self.use_cache:
            if exists(aux_path):
                with open(aux_path) as reader:
                    content = json.load(reader)
                    self.ids_map[split] = content['ids_map']
                
            else:
                with open(aux_path, 'w') as writer:
                    content = {'ids_map': self.ids_map[split]}
                    json.dump(content, writer)
    
    @property
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.tensors['predict'],
            batch_size=self.batch_size,
            num_workers=self.num_workers)
    
    @abstractmethod
    def load(self, split: str) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def process(self, split: str, samples: dict) -> Tuple[TensorDataset, list]:
        raise NotImplementedError


class GMDatasetLMMixin(GMDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer: PreTrainedTokenizer = ...
        self.tokenizer_cls: Type[AutoTokenizer] = ...

        self.init_tokenizer()

        self.cls_id = self.tokenizer.cls_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id

        self.texts = {'train': None, 'val': None, 'test': None, 'predict': None}
        self.offsets = {'train': None, 'val': None, 'test': None, 'predict': None}

    def init_tokenizer(self) -> None:
        self.init_tokenizer_cls()

        if self.fork is not None:
            tokenizer_path = join(self.fork, 'tokenizer')
            self.tokenizer = self.tokenizer_cls.from_pretrained(tokenizer_path)
            print(f'Loaded tokenizer from: {tokenizer_path}')

        elif exists(join(self.model_dir, 'tokenizer')):
            tokenizer_path = join(self.model_dir, 'tokenizer')
            self.tokenizer = self.tokenizer_cls.from_pretrained(tokenizer_path)
            print(f'Loaded tokenizer from: {tokenizer_path}')
        
        else:
            self.tokenizer = self.tokenizer_cls.from_pretrained(self.model_name)
    
    def init_tokenizer_cls(self) -> None:
        self.tokenizer_cls = AutoTokenizer

    def _setup_aux(self, split: str) -> None:
        aux_path = join(self.cache_dir, f'{split}.json')

        if self.use_cache:
            if exists(aux_path):
                with open(aux_path) as reader:
                    content = json.load(reader)
                    self.ids_map[split] = content['ids_map']
                    self.texts[split] = content['texts']
                    self.offsets[split] = content['offsets']
                
            else:
                with open(aux_path, 'w') as writer:
                    content = {
                        'ids_map': self.ids_map[split],
                        'texts': self.texts[split],
                        'offsets': self.offsets[split]}

                    json.dump(content, writer)
