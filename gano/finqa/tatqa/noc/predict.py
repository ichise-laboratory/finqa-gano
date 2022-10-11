from gano.finqa.tatqa.noc.experiments import NocExperiment
from gano.finqa.tatqa.tagop.predict import TagOpPredictor


class NocPredictor(NocExperiment, TagOpPredictor):
    pass


if __name__ == '__main__':
    predictor = NocPredictor()
    predictor.predict()
