import re


def patch_single_unit(table: dict, rules: dict) -> None:
    if len(rules['unit']) == 1:
        u = rules['unit'][0]
        
        for i in range(len(table[0])):
            table[u[0]][i] = table[u[0]][u[1]]
        

def merge_table_header(table: dict, rules: dict) -> None:
    rlen, clen = len(table), len(table[0])
    mtbl = [['n'] * clen for _ in range(rlen)]
    merged = [[False] * clen for _ in range(rlen)]

    for c, t in zip(('l', 'r'), ('left', 'right')):
        for m in rules[f'merge_{t}']:
            if len(table[m[0]][m[1]]):
                mtbl[m[0]][m[1]] = f'{c}f'
            else:
                mtbl[m[0]][m[1]] = c
    
    for m in rules['merge_row']:
        mtbl[m[0]][m[1]] = 'o'
    
    for i in range(rlen):
        for j in range(clen):
            if mtbl[i][j] == 'lf':
                k = j - 1

                while k >= 0 and mtbl[i][k] == 'l':
                    table[i][k] = table[i][j]
                    k += 1
            
            elif mtbl[i][j] == 'rf':
                k = j + 1

                while k < clen and mtbl[i][k] == 'r':
                    table[i][k] = table[i][j]
                    k += 1
            
            elif mtbl[i][j] == 'o' and not merged[i][j]:
                k = j

                while k < clen and mtbl[i][k] == 'o':
                    merged[i][k] = True
                    k += 1
                
                cell = re.sub(r' +', ' ', ' '.join(table[i][j:k])).strip(' ')

                for l in range(j, k):
                    table[i][l] = cell
    
    return table
