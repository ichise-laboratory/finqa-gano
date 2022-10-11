from gano.finqa.tatqa.geer.experiments import GeerExperiment
from gano.finqa.tatqa.tagop.predict import TagOpPredictor


class GeerPredictor(GeerExperiment, TagOpPredictor):
    def add_args(self, defaults: dict = None) -> None:
        super().add_args(defaults)

        add = lambda *args, **kwargs: self.parser.add_argument(*args, **kwargs)
        add('--debug-graph', type=str, default=None, help='Output graph(s) for Gephi')
        add('--gnn', type=str, default='sage', help='GNN algorithm')
        add('--ignore-empty-cells', action='store_true', help='Do not fill empty table cells with <empty>')

    def build_dataset_params(self) -> dict:
        params = super().build_dataset_params()
        params['debug_graph'] = self.args.debug_graph
        params['fill_empty_cells'] = not self.args.ignore_empty_cells
        return params
    
    def build_model_params(self) -> dict:
        return {
            'gnn_cls': self.args.gnn,
            **super().build_model_params()}


if __name__ == '__main__':
    predictor = GeerPredictor()
    predictor.predict()
