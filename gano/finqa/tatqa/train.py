from gano.finqa.train import FinQATrainer, FinQATrainerLMMixin


class TatQATrainer(FinQATrainer):
    def add_args(self, defaults: dict = None) -> None:
        super().add_args(self.update_args(defaults, {
            'template': 'f.b.e.l.i'}))
        
        add = lambda *args, **kwargs: self.parser.add_argument(*args, **kwargs)
        add('--sample-ratio', type=float, default=None, help='Train with part of the training data')
    
    def build_dataset_params(self) -> dict:
        return {
            'sample_ratio': self.args.sample_ratio,
            **super().build_dataset_params()}


class TatQATrainerForLM(TatQATrainer, FinQATrainerLMMixin):
    pass
