import json
import re

from gano.finqa.tatqa.geer.utils import merge_table_header as merge

MONTHS = (
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december')

UNITS = ('thousand', 'million')


def locate_headers(samples: list, split: str, debug: bool = False):
    with open('tmp/tatqa/analyses/tables/skip.json') as reader:
        skips = set(json.load(reader)[split])

    incorrect = 0

    for sample in samples:
        tid = sample['table']['uid']
        table = sample['table']['table']
        head_limit, cand_limit = None, None
        col_fills = [0] * len(table[0])

        for i, row in enumerate(table):
            for j, cell in enumerate(row):
                if len(cell) > 0:
                    col_fills[j] = 1
            
            if sum(col_fills) < len(row) - 1:
                continue

            elif cand_limit is None:
                cand_limit = i + 1

            if i == 0:
                continue

            has_numbers = has_years = has_months = False
            has_units = has_chars = empty = False
            has_scales = has_dash = has_dollar = False

            for j, cell in enumerate(row):
                cell = cell.lower()

                if j > 0 and re.search(r'[0-9]', cell):
                    has_numbers = True
                
                if j > 0 and re.search(r'[0-9][0-9][0-9][0-9]', cell):
                    has_years = True

                if j > 0 and any(m in cell for m in MONTHS):
                    has_months = True
                
                if any(u in cell for u in UNITS):
                    has_units = True

                if j > 1 and len(re.findall(r'[a-z]', cell)) > 1:
                    has_chars = True
                
                if '000' in cell and ',000' not in cell:
                    has_scales = True
                
                if j > 0 and '$' in cell:
                    has_dollar = True

                if j > 0 and cell == '-':
                    has_dash = True
                
            if all(len(row[i]) == 0 for i in range(1, len(row))):
                empty = True
            
            if head_limit is None:
                if not has_numbers and has_units:
                    continue
            
                if has_chars and not has_dollar:
                    continue
                    
                if has_scales:
                    continue

                if has_numbers and not has_years and not has_months:
                    head_limit = i
                
                if empty:
                    head_limit = i

                if has_dash:
                    head_limit = i
        
        if head_limit is None:
            head_limit = cand_limit or 1

        if split != 'test':
            annotation = sample['table']['annotation']
            
            if tid not in skips and head_limit != len(annotation['header']):
                incorrect += 1

                if debug:
                    print(tid, ':', len(annotation['header']), '|', head_limit)
        
            sample['table']['pred'] = {'header': [i for i in range(head_limit)]}
        
        else:
            sample['table']['annotation'] = {'header': [i for i in range(head_limit)]}
    
    print('Head row size incorrect:', incorrect, 'of', len(samples))
    exit()

    return samples


def merge_headers(samples: list, debug: bool = False) -> list:
    incorrect = 0
    skip_keys = [
        'as of', 'year', 'quarter', 'month', 
        'december', 'october', 'september', 'august',
        'thousand', 'million']

    for sample in samples:
        table = sample['table']['table']
        annotation = sample['table']['annotation']
        label = []
    
        for i in range(len(table)):
            label.append([table[i][j] for j in range(len(table[i]))])

        label = merge(label, annotation)

        for i in range(0, len(sample['table']['pred']['header']) - 1):
            row = table[i]

            for j in range(1, len(row)):
                if len(row[j]) == 0:
                    k = j - 1

                    while k > 0 and len(row[k]) == 0:
                        k -= 1
                    
                    if k > 0:
                        row[j] = row[k]
                
                if len(row[j]) == 0:
                    k = j + 1

                    while k < len(row) and len(row[k]) == 0:
                        k += 1
                    
                    if k < len(row):
                        row[j] = row[k]
        
        sample_incorrect = False

        for i in range(len(table)):
            for j in range(1, len(table[0])):
                if table[i][j].strip() != label[i][j].strip():
                    pred = table[i][j].strip().lower()
                    gold = label[i][j].strip().lower()

                    if any(s in pred or s in gold for s in skip_keys):
                        pass

                    else:
                        sample_incorrect = True

                        if debug:
                            print('|' + table[i][j].strip() + '|' + label[i][j].strip() + '|')
        
        if sample_incorrect:
            incorrect += 1

            if debug:
                print(sample['table']['uid'])

                for i in range(0, len(annotation['header']) - 1):
                    print(' | '.join([c.strip() for c in label[i]]))
                
                print('---')

                for i in range(0, len(annotation['header']) - 1):
                    print(' | '.join([c.strip() for c in table[i]]))
                
                input()
        
        del sample['table']['annotation']
        sample['table']['annotation'] = {'header': sample['table']['pred']['header']}
        del sample['table']['pred']
        
    print('Merge incorrect:', incorrect)
    return samples


def main():
    for split in ('train', 'val'):
        with open(f'data/finqa/tatqa/tagop/annotated/{split}-m.json') as reader:
            samples = json.load(reader)
        
        samples = locate_headers(samples, split)
        samples = merge_headers(samples)

        with open(f'data/finqa/tatqa/tagop/annotated/{split}-ma.json', 'w') as writer:
            json.dump(samples, writer, indent=2)
    
    with open('data/tatqa/raw/tatqa_dataset_test.json') as reader:
        samples = json.load(reader)
    
    samples = locate_headers(samples, 'test')

    with open(f'data/finqa/tatqa/tagop/annotated/test-ma.json', 'w') as writer:
        json.dump(samples, writer, indent=2)


if __name__ == '__main__':
    main()
