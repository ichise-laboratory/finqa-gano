from gano.finqa.tatqa.noc.experiments import NocExperiment
from gano.finqa.tatqa.tagop.test import TagOpTester


class NocTester(NocExperiment, TagOpTester):
    pass


if __name__ == '__main__':
    tester = NocTester()
    tester.test()
