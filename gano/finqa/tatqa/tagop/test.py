from gano.finqa.tatqa.tagop.experiments import TagOpExperiment
from gano.finqa.tatqa.test import TatQATesterForLM


class TagOpTester(TagOpExperiment, TatQATesterForLM):
    def add_args(self, defaults: dict = None) -> None:
        super().add_args(defaults)

        get = lambda key, default: defaults[key] if defaults is not None and key in defaults else default
        add = lambda *args, **kwargs: self.parser.add_argument(*args, **kwargs)
        add('--fill-empty-cells', action='store_true', help='Fill empty table cells with <empty>')
        add('--paragraphs', type=int, default=get('paragraphs', None), help='Number (limit) of (sorted) paragraphs')
        add('--scheme', type=str, default=get('scheme', 'io'), help='Token classification scheme')
        add('--use-lstm', action='store_true', help='Use LSTM for text prediction')
    
    def build_dataset_params(self) -> dict:
        return {
            'fill_empty_cells': self.args.fill_empty_cells,
            'n_paragraphs': self.args.paragraphs,
            'scheme': self.args.scheme,
            **super().build_dataset_params()}
    
    def build_model_params(self) -> dict:
        return {
            'scheme': self.args.scheme,
            'use_lstm': self.args.use_lstm,
            **super().build_model_params()}


if __name__ == '__main__':
    tester = TagOpTester()
    tester.test()
