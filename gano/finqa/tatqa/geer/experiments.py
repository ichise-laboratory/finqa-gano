from gano.finqa.tatqa.geer.datasets import GeerDataset
from gano.finqa.tatqa.geer.models import GeerLightning
from gano.finqa.tatqa.tagop.experiments import TagOpExperiment


class GeerExperiment(TagOpExperiment):
    PIPELINE = 'g:geer'

    def dataset_cls(self, model: str):
        if model in self.MODELS.keys(): return GeerDataset
        else: self.invalid_model(model)
    
    def model_cls(self, model: str):
        if model in self.MODELS.keys(): return GeerLightning
        else: self.invalid_model(model)
