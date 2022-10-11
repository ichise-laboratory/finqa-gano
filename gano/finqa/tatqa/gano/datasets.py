import logging

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gano.finqa.tatqa.geer.datasets import GeerDataset, GeerTensorDataset
from gano.finqa.tatqa.noc.datasets import NocDataset


class GanoDataset(GeerDataset, NocDataset):
    def _process_tensors(self, inputs: list, samples: list) -> Dataset:
        features = []

        logging.info('Building graph tensors.')

        for n in tqdm(inputs):
            n.graph = self._process_graph(n, samples[n.s['idx']])

            features.append((
                torch.tensor(n.s['idx']),

                torch.tensor(n.t['mask']),
                torch.tensor(n.t['nums'], dtype=torch.float),
                torch.tensor(n.t['cell_idx']),
                torch.tensor(n.t['nums_map']),

                torch.tensor(n.p['mask']),
                torch.tensor(n.p['nums'], dtype=torch.float),
                torch.tensor(n.p['word_idx']),
                torch.tensor(n.p['nums_map']),
                
                torch.tensor(n.s['tags']),
                torch.tensor(n.s['opr_tag']),

                torch.tensor(n.o['tags']),
                torch.tensor(n.o['nums'], dtype=torch.float),
                torch.tensor(n.o['mask']),

                torch.tensor(n.s['scale_tag']),
                torch.tensor(self._process_types(n.s['types'])),

                torch.tensor(n.d['nums'], dtype=torch.float),
                torch.tensor(n.d['pos']),

                n.graph,

                torch.tensor(n.s['input_ids'], dtype=torch.int),
                torch.tensor(n.s['attention_mask'], dtype=torch.int)))
        
        return GeerTensorDataset(features)


class GanoFinBertDataset(GanoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_token_len = 512
