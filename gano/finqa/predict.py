from gano.manage.predict import GMPredictor, GMPredictorLMMixin


class FinQAPredictor(GMPredictor):
    pass


class FinQAPredictorLMMixin(FinQAPredictor, GMPredictorLMMixin):
    pass
