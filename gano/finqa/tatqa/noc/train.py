from gano.finqa.tatqa.noc.experiments import NocExperiment
from gano.finqa.tatqa.tagop.train import TagOpTrainer


class NocTrainer(NocExperiment, TagOpTrainer):
    pass


if __name__ == '__main__':
    trainer = NocTrainer()
    trainer.train()
