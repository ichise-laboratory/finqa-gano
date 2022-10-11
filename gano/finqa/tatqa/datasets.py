import json
import logging
from os.path import join

from gano.finqa.datasets import FinQADataset, FinQADatasetLMMixin


class TatQADataset(FinQADataset):
    def __init__(self, sample_ratio: float = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_ratio = sample_ratio

    def load(self, split: str):
        if split == 'predict': 
            path = join(self.data_dir, 'raw/tatqa_dataset_test.json')
        
        elif split == 'train':
            path = join(self.data_dir, 'tagop/annotated/train-bv.json')
        
        elif split == 'val':
            path = join(self.data_dir, 'tagop/annotated/val-bv.json')
        
        elif split == 'test':
            path = join(self.data_dir, 'tagop/annotated/val-bv.json')

        with open(path, encoding='utf-8') as reader:
            collection = json.load(reader)
        
        samples = []

        for doc in collection:
            table = doc['table']
            paragraphs = doc['paragraphs']
            
            for question in doc['questions']:
                samples.append({
                    'table': table,
                    'paragraphs': paragraphs,
                    'question': question})
        
        if self.sample_ratio is not None:
            if split == 'train':
                logging.info('Setting sample ratio to %.3f' % self.sample_ratio)
                path = join(self.data_dir, 'tagop/reduced/train-bv-%.3f.json' % self.sample_ratio)
                filtered = []

                with open(path, encoding='utf-8') as reader:
                    qids = set(json.load(reader))
            
                for sample in samples:
                    if sample['question']['uid'] in qids:
                        filtered.append(sample)

                samples = filtered
                
        logging.info(f'Loaded data from: {path} ({len(samples)} documents).')
        return samples


class TatQADatasetForLM(TatQADataset, FinQADatasetLMMixin):
    pass
