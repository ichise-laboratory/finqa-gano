from gano.manage.datasets import GMDataset, GMDatasetLMMixin


class FinQADataset(GMDataset):
    def preprocess(self, split: str, samples: list) -> list:
        for sample in samples:
            table = sample['table']['table']

            for row_i in range(len(table)):
                for col_i in range(len(table[row_i])):
                    if isinstance(table[row_i][col_i], str):
                        table[row_i][col_i] = {'text': table[row_i][col_i]}
        
        return samples


class FinQADatasetLMMixin(FinQADataset, GMDatasetLMMixin):
    pass
