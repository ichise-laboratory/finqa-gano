from gano.finqa.experiments import FinQAExperiment, FinQAExperimentForLM


class TatQAExperiment(FinQAExperiment):
    PROJECT = 'tatqa'


class FinQAExperimentForLM(TatQAExperiment, FinQAExperimentForLM):
    pass
