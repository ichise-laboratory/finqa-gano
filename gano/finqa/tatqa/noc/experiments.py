from gano.finqa.tatqa.noc.datasets import NocDataset
from gano.finqa.tatqa.noc.models import NocLightning
from gano.finqa.tatqa.tagop.experiments import TagOpExperiment


class NocExperiment(TagOpExperiment):
    PIPELINE: str = 'no:noc'

    def dataset_cls(self, model: str):
        if model in self.MODELS.keys(): return NocDataset
        else: self.invalid_model(model)
    
    def model_cls(self, model: str):
        if model in self.MODELS.keys(): return NocLightning
        else: self.invalid_model(model)
