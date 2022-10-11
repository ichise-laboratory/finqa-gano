from gano.finqa.tatqa.experiments import FinQAExperiment, FinQAExperimentForLM
from gano.finqa.tatqa.tagop.datasets import TagOpDataset, TagOpFinBertDataset
from gano.finqa.tatqa.tagop.models import TagOpLightning


class TagOpExperiment(FinQAExperimentForLM, FinQAExperiment):
    PIPELINE = 't:tagop'

    MODELS = {
        'bb': 'bert-base-uncased',
        'bl': 'bert-large-uncased',
        'db': 'distilbert-base-uncased',
        'fb': 'yiyanghkust/finbert-pretrain',
        'mb': 'google/mobilebert-uncased',
        'rb': 'roberta-base',
        'rl': 'roberta-large',
        'sb': 'nlpaueb/sec-bert-base',
        'sn': 'nlpaueb/sec-bert-num',
        'sp': 'nlpaueb/sec-bert-shape',
        'tb': 'huawei-noah/TinyBERT_General_4L_312D'}
    
    def dataset_cls(self, model: str):
        if model == 'fb': return TagOpFinBertDataset
        elif model in self.MODELS.keys(): return TagOpDataset
        else: self.invalid_model(model)
    
    def model_cls(self, model: str):
        if model in self.MODELS.keys(): return TagOpLightning
        else: self.invalid_model(model)
