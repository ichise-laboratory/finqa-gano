from gano.manage.train import GMTrainer, GMTrainerLMMixin


class FinQATrainer(GMTrainer):
    pass


class FinQATrainerLMMixin(FinQATrainer, GMTrainerLMMixin):
    pass
