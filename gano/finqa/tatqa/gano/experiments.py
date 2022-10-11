from gano.finqa.tatqa.gano.datasets import GanoDataset, GanoFinBertDataset
from gano.finqa.tatqa.gano.models import GanoLightning
from gano.finqa.tatqa.geer.experiments import GeerExperiment
from gano.finqa.tatqa.noc.experiments import NocExperiment


class GanoExperiment(GeerExperiment, NocExperiment):
    PIPELINE = 'gn:gano'

    def dataset_cls(self, model: str):
        if model == 'fb': return GanoFinBertDataset
        elif model in self.MODELS.keys(): return GanoDataset
        else: self.invalid_model(model)
    
    def model_cls(self, model: str):
        if model in self.MODELS.keys(): return GanoLightning
        else: self.invalid_model(model)
