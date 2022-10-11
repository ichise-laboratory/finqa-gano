from gano.finqa.test import FinQATester, FinQATesterLMMixin


class TatQATester(FinQATester):
    pass


class TatQATesterForLM(TatQATester, FinQATesterLMMixin):
    pass
