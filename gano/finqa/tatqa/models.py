import json

from gano.finqa.models import FinQALightning, FinQALightningLMMixin
from gano.finqa.tatqa.metrics import TaTQAEmAndF1


class TatQALightning(FinQALightning):
    RESULT_KEYS = ('em', 'f1', 'scale', 'opr')
    RESULT_TYPES = ('arithmetic', 'count', 'multi-span', 'span')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {'train': TaTQAEmAndF1(), 'val': TaTQAEmAndF1(), 'test': TaTQAEmAndF1()}
    
    def compute_metric(self, metric: TaTQAEmAndF1, reset: bool = True) -> tuple:
        detail_em, detail_f1 = metric.get_detail_metric()
        em, f1, scale, opr = metric.get_overall_metric()

        result = {
            'em': em, 'f1': f1, 
            'scale': scale, 'opr': opr,
            **self._extract_metric('em', detail_em),
            **self._extract_metric('f1', detail_f1)}
        
        if reset:
            metric.reset()
        
        return result
    
    def _extract_metric(self, mtype: str, detail: dict) -> dict:
        result = {}
        headers = {
            'tbl': f"('{mtype}', 'table')", 
            'hyb': f"('{mtype}', 'table-text')", 
            'prg': f"('{mtype}', 'text')"}
        detail = json.loads(detail.to_json())

        for metric in self.RESULT_TYPES:
            for key, col in headers.items():
                if metric in detail[col]:
                    result[f'{mtype}.{metric}.{key}'] = detail[col][metric]

        return result


class TatQALightningForLM(TatQALightning, FinQALightningLMMixin):
    pass
