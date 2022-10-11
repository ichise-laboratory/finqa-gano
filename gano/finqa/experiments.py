from gano.manage.experiments import GMExperiment, GMExperimentLMMixin


class FinQAExperiment(GMExperiment):
    DOMAIN = 'finqa'


class FinQAExperimentForLM(FinQAExperiment, GMExperimentLMMixin):
    pass
