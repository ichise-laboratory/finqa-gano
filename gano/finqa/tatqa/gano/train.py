from gano.finqa.tatqa.gano.experiments import GanoExperiment
from gano.finqa.tatqa.geer.train import GeerTrainer
from gano.finqa.tatqa.noc.train import NocTrainer


class GanoTrainer(GanoExperiment, GeerTrainer, NocTrainer):
    pass


if __name__ == '__main__':
    trainer = GanoTrainer()
    trainer.train()
