from gano.finqa.tatqa.gano.experiments import GanoExperiment
from gano.finqa.tatqa.geer.test import GeerTester
from gano.finqa.tatqa.noc.test import NocTester


class GanoTester(GanoExperiment, GeerTester, NocTester):
    pass


if __name__ == '__main__':
    tester = GanoTester()
    tester.test()
