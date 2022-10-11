import json
import logging
import pathlib
from os.path import join

import networkx as nx
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from gano.finqa.tatqa.tagop.datasets import TagOpDataset
from gano.finqa.tatqa.geer.utils import merge_table_header as merge

torch.multiprocessing.set_sharing_strategy('file_system')


class GeerTensorDataset(Dataset):
    def __init__(self, features: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = features
    
    def __getitem__(self, index: int):
        return self.features[index]

    def __len__(self):
        return len(self.features)


class GeerDataset(TagOpDataset):
    def __init__(self, debug_graph: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.graphs = {'train': None, 'val': None, 'test': None, 'predict': None}
        self.debug_graph = debug_graph
    
    def load(self, split: str):
        samples = super().load(split)

        if split in ('train', 'val'):
            path = join(self.data_dir, f'tagop/annotated/{split}-m.json')

        elif split == 'test':
            path = join(self.data_dir, f'tagop/annotated/val-ma.json')

        elif split == 'predict':
            path = join(self.data_dir, f'tagop/annotated/test-ma.json')

        with open(path) as reader:
            collection = json.load(reader)
            
        logging.info(f'Loaded graph annotation from: {path}')
        
        rules = {}

        for doc in collection:
            rules[doc['table']['uid']] = {
                **doc['table']['annotation'], 'applied': False}
        
        for sample in samples:
            table = sample['table']
            table['header'] = rules[table['uid']]

        return samples

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
    
    def test_dataloader(self):
        return DataLoader(
            self.tensors['test'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers)
    
    @property
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.tensors['predict'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def _debug_graph(self, G: nx.Graph, sample: dict) -> None:
        if self.debug_graph in ('all', 'interactive') \
            or self.debug_graph == sample['question']['uid']:

            H = G.copy()
            degs = H.degree()
            dots = [n for (n, deg) in degs if deg == 0]
            H.remove_nodes_from(dots)

            output = join(self.output_dir, 'graphs')
            path = join(output, f"{sample['question']['uid']}.gexf")
            pathlib.Path(output).mkdir(parents=True, exist_ok=True)
            nx.write_gexf(H, path)

            if self.debug_graph == 'interactive':
                print('Saved to:', path)
                input()
    
    def _process_graph(self, n: tuple, sample: dict) -> Data:
        def link(source: list, target: list) -> list:
            edges = []

            for s in source:
                for t in target:
                    edges.append((s, t))
            
            return edges

        G = nx.DiGraph()

        tokens = self.tokenizer.convert_ids_to_tokens(n.s['input_ids'])
        nodes = [(i, {'token': tokens[i]}) for i in range(len(tokens))]
        edges = []

        theads = sample['table']['header']['header']
        qrange = range(1, n.s['lens'][0] - 1)
        trange = range(n.s['lens'][0], n.s['lens'][0] + n.s['lens'][1] - 1)
        rlen = len(sample['table']['table'])
        clen = len(sample['table']['table'][0])
        tpos = []

        for i in range(rlen):
            tpos.append([[] for _ in range(clen)]) 
        
        for i in trange:
            pos = n.t['cell_pos'][i]

            if pos is not None:
                tpos[pos[0]][pos[1]].append(i)
        
        for i in range(rlen):
            for j in range(clen):
                if i in theads:
                    edges += link(list(qrange), tpos[i][j])

                    if i > 0 and j > 0:
                        for k in range(1, clen):
                            if len(tpos[i - 1][k]) > 0:
                                edges += link(tpos[i - 1][k], tpos[i][j])
                
                elif j == 0:
                    edges += link(list(qrange), tpos[i][j])
                    
                    if i > 0:
                        k = i - 1

                        while k > 0 and len(tpos[k][j]) == 0:
                            k -= 1
                        
                        edges += link(tpos[k][j], tpos[i][j])
                
                else:
                    k = i - 1

                    while k > 0 and (k not in theads or len(tpos[k][j]) == 0):
                        k -= 1
                    
                    edges += link(tpos[k][j], tpos[i][j])
                    edges += link(tpos[i][0], tpos[i][j])

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        self._debug_graph(G, sample)
        return from_networkx(G)

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

                torch.tensor(n.p['mask']),
                torch.tensor(n.p['nums'], dtype=torch.float),
                torch.tensor(n.p['word_idx']),
                
                torch.tensor(n.s['tags']),
                torch.tensor(n.s['opr_tag']),
                torch.tensor(n.s['order_tag']),
                torch.tensor(n.s['scale_tag']),
                torch.tensor(self._process_types(n.s['types'])),

                n.graph,

                torch.tensor(n.s['input_ids'], dtype=torch.int),
                torch.tensor(n.s['attention_mask'], dtype=torch.int)))
        
        return GeerTensorDataset(features)
    
    def _process_types(self, types: list) -> list:
        type_lens = [0, 0, 0]

        for t in types:
            type_lens[t - 1] += 1
        
        return type_lens
