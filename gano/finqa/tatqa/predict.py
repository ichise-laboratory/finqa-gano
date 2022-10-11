from gano.finqa.predict import FinQAPredictor, FinQAPredictorLMMixin


class TatQAPredictor(FinQAPredictor):
    pass


class TatQAPredictorForLM(TatQAPredictor, FinQAPredictorLMMixin):
    pass
