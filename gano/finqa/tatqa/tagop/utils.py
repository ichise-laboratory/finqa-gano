
import math
import re
import string

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from gano.finqa.tatqa.tagop.configs import OPERATORS, SCALES

EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"
MONTHS = ('january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december')


def convert_offsets(offsets: list, idx: list):
    merged = [None] * (max(idx) + 1)

    for i in range(len(idx)):
        if idx[i] > 0:
            if merged[idx[i]] is None:
                merged[idx[i]] = list(offsets[i])
            else:
                merged[idx[i]][1] = offsets[i][1]
    
    for i in range(len(merged)):
        if merged[i] is None:
            merged[i] = [0, 0]

    return [tuple(m) for m in merged]


def get_span(pred: list, scores: list, word_idx: list, offsets: list, text: str) -> list:
    spans, in_span = [], False
    offsets = convert_offsets(offsets, word_idx)

    for i in range(len(pred)):
        if pred[i] != 0 and not in_span:
            in_span = True
            start_i = i
        
        if pred[i] == 0 and in_span:
            in_span = False
            spans.append((start_i, i))

    if in_span:
        spans.append((start_i, len(pred)))

    best_score, best_span = float('-inf'), None

    for span in spans:
        score = np.mean(scores[span[0]:span[1]])

        if score > best_score:
            best_score = score
            best_span = span
    
    if best_span is None:
        return []  
    else:
        offset = (offsets[best_span[0]][0], offsets[best_span[1] - 1][1])
        answer = text[offset[0]:offset[1]]
        return [answer]


def get_qst_spans(pred: list, offsets: list, text: str) -> list:
    answers = []
    i = 0

    while i < len(pred):
        if pred[i] > 0:
            j = i

            while j < len(pred) and pred[j] > 0:
                j += 1
            
            offset = (offsets[i][0], offsets[j - 1][1])
            answers.append(text[offset[0]:offset[1]])
            i = j
        
        else:
            i += 1
    
    return answers


def get_tbl_spans(pred: list, offsets: list, cell_idx: list, text: str) -> list:
    answers = []
    offsets = convert_offsets(offsets, cell_idx)
    
    for i in range(len(pred)):
        if pred[i] != 0:
            offset = (offsets[i][0], offsets[i][1])
            answer = text[offset[0]:offset[1]]

            if len(answer) and answer not in answers:
                answers.append(answer)
    
    return answers


def get_prg_spans(pred: list, offsets: list, word_idx: list, text: str) -> list:
    spans, answers, in_span = [], [], False
    offsets = convert_offsets(offsets, word_idx)

    for i in range(len(pred)):
        if pred[i] != 0 and not in_span:
            in_span = True
            start_i = i
        
        if pred[i] == 0 and in_span:
            in_span = False
            spans.append((start_i, i))

    if in_span:
        spans.append((start_i, len(pred)))
    
    for span in spans:
        offset = (offsets[span[0]][0], offsets[span[1] - 1][1])
        answer = text[offset[0]:offset[1]]

        if len(answer) and answer not in answers:
            answers.append(answer)
    
    return answers


def get_nums(pred: list, nums: list) -> list:
    answers, pos = [], []

    for i in range(1, len(pred)):
        if pred[i] != 0 and not np.isnan(nums[i - 1]):
            answers.append(nums[i - 1])
            pos.append(i)
    
    return answers, pos


def _clean_num(text:str):
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def _extract_one_num_from_str(s):
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)

    if len(groups) == 0:
        return None

    num = groups[0][0]

    if num == '':
        return None

    if '.' in num:
        return float(num)

    return int(num)


def scale_to_num(scale):
    scale = scale.lower()
    num = 1

    if 'hundred' in scale:
        num = 100
    elif 'thousand' in scale:
        num = 1000
    elif 'million' in scale:
        num = 1000000
    elif 'billion' in scale:
        num = 1000000000
    elif 'percent' in scale:
        num = 0.01

    return num


def _word_scale_handle(x):
    """
    :param x: 1 million = 1,000,000
    :return:
    """
    iter = re.finditer('([\d.]+\s?[a-zA-Z]+)', x)

    for one in iter:
        text = one.group(0).lower()
        scale_val = scale_to_num(text)
        return scale_val

    return 1


def _negative_num_handle(x):
    """
    :param x:  transform (134) -> -134
    :return:
    """
    all = re.findall('(\([\d.\s]+\))', x.strip())

    if len(all) > 0:
        return -1

    return 1


def _percent_num_handle(x):
    """
    :param x:  transform 12% -> 12/100
    :return:
    """
    all = re.findall('([\d.\s]+%)', x.strip())

    if len(all) > 0:
        return 0.01

    return 1


def is_number(text: str) -> bool:
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()

        if len(words) == 0:
            """1023 or 1 million"""
            return False

        num = float(words[0])

        if np.isnan(num):
            return False

        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False

        return True

    except ValueError:
        return False


def _cell_is_num(text: str) -> bool:
    return not re.search(r'[^(][a-zA-Z]+[^)]', text.replace(' ', ''))


def _cell_remove_remark(text: str) -> str:
    if '(' in text:
        if text.startswith('('):
            return text

        elif re.search(r'^[$£<>]\(', text):
            return text

        else:
            if len(re.findall(r'[a-zA-Z]', text)):
                return text
            
            else:
                return text[:text.index('(')].strip()
    
    return text


def _cell_is_negative(text: str, rhead: str, query: str) -> int:
    keywords = ('expense', 'deferred', 'decrease', 'impairment', 'payment', 
        'forfeit', 'loss', 'liabilit', 'expire', 'cost', 'write-off', 'depreciation', 
        'deficit', 'benefit', 'divestment', 'allowance', 'amortization',
        'divestment', 'debt', 'unearn')
    
    if re.search(r'^[$£<>]*\(', text):
        if any(k in query for k in keywords):
            return 1
        
        elif any(k in query and k in rhead for k in keywords):
            return 1

        else:
            return -1
    
    return 1


def _cell_is_zero(text: str) -> bool:
    dash_codes = (
        '-', u'\u2013', u'\u2014', 
        '$-', u'$\u2013', u'$\u2014',
        '-%', u'\u2013%', u'\u2014%')

    return text.replace(' ', '') in dash_codes


def to_number(
    text: str, 
    table: dict = None, 
    row: int = None, 
    query: str = '',
    number: bool = False) -> float:

    if table is not None:
        if not _cell_is_num(text):
            return None

        cell = _cell_remove_remark(text.replace(' ', ''))
        num = _extract_one_num_from_str(cell)
        negative_flag = _cell_is_negative(cell, 
            table[row][0]['text'].lower(), query.lower())

        if _cell_is_zero(text):
            num = 0

    else:
        num = _extract_one_num_from_str(text)
        negative_flag = _negative_num_handle(text)

    scale_val = _word_scale_handle(text)
    percent_flag = _percent_num_handle(text)

    if num is not None:
        if not number and num >= 1900 and num <= 2100 and str(num) in text:
            return None
        else:
            return round(num * scale_val * negative_flag * percent_flag, 4)

    return None


def facts_to_nums(facts):
    return [to_number(f) for f in facts]


def _get_operators(derivation:str):
    res = []

    for c in derivation:
        if c in ['+', '-', '*', '/']:
            res.append(c)

    return res


def is_average(num_facts:list, answer: float):
    return round(np.average(num_facts), 2) == round(answer, 2)


def is_change_ratio(num_facts:list, answer: float, return_order: bool = False):
    if len(num_facts) != 2:
        return False

    cands = []

    if num_facts[1] != 0:
        ori_percent = round(100 * (num_facts[0] - num_facts[1]) / num_facts[1], 2)
        cands.append(ori_percent)
    if num_facts[0] != 0:
        ori_percent = round(100 * (num_facts[1] - num_facts[0]) / num_facts[0], 2)
        cands.append(ori_percent)
    
    valid = round(answer, 2) in cands

    if valid and return_order:
        if round(answer, 2) == cands[0]:
            return 'normal'
        else:
            return 'reverse'
    
    else:
        return valid


def is_division(num_facts:list, answer: float, return_order: bool = False):
    if len(num_facts) != 2:
        return False

    cands = []

    if num_facts[1] != 0:
        cands.append(round(num_facts[0]/num_facts[1], 2))
        cands.append(100 * round(num_facts[0]/num_facts[1], 2))
    if num_facts[0] != 0:
        cands.append(round(num_facts[1]/num_facts[0], 2))
        cands.append(100 * round(num_facts[1]/num_facts[0], 2))
    
    valid = round(answer, 2) in cands

    if valid and return_order:
        if round(answer, 2) in cands[:2]:
            return 'normal'
        else:
            return 'reverse'
    
    else:
        return valid


def is_diff(num_facts:list, answer: float, return_order: bool = False):
    if len(num_facts) != 2:
        return False

    ans_1 = round(num_facts[0] - num_facts[1], 2)
    ans_2 = round(num_facts[1] - num_facts[0], 2)

    valid = round(answer, 2) in (ans_1, ans_2)

    if valid and return_order:
        if round(answer, 2) == ans_1:
            return 'normal'
        else:
            return 'reverse'
    
    else:
        return valid


def is_sum(num_facts:list, answer: str) -> bool:
    return round(np.sum(num_facts), 2) == round(answer, 2)


def is_multiply(num_facts:list, answer: str) -> bool:
    return round(np.prod(num_facts), 2) == round(answer, 2)


def extract_operator(derivation:str, answer_type:str, facts:list, 
    answer: str, mapping:dict, scale) -> str:
    
    if answer_type == 'span':
        if 'table' in mapping:
            return 'span-table'
        else:
            return 'span-text'

    elif answer_type == 'multi-span':
        return 'multi-span'
    
    elif answer_type == 'count':
        return 'count'
    
    elif answer_type == 'arithmetic':
        num_facts = facts_to_nums(facts)

        if not is_number(str(answer)):
            return None
        elif is_change_ratio(num_facts, answer):
            return 'change-ratio'
        elif is_average(num_facts, answer):
            return 'average'
        elif is_sum(num_facts, answer):
            return 'sum'
        elif is_multiply(num_facts, answer):
            return 'multiply'
        elif is_diff(num_facts, answer):
            return 'diff'
        elif is_division(num_facts, answer):
            return 'divide'
        
        operators = _get_operators(derivation)

        if len(operators) == 1:
            if operators[0] == '/':
                return 'divide'
            elif operators[0] == '-':
                return 'diff'
            elif operators[0] == '*':
                return 'multiply'
            elif operators[0] == '+':
                return 'sum'
    
    return None

def extract_order(operator: str, facts:list, answer: str):
    num_facts = facts_to_nums(facts)

    if operator == 'change-ratio':
        return is_change_ratio(num_facts, answer, return_order=True)
    elif operator == 'divide':
        return is_division(num_facts, answer, return_order=True)
    elif operator == 'diff':
        return is_diff(num_facts, answer, return_order=True)
    
    return None


def white_space_fix(text: str) -> str:
    return ' '.join(text.split())


def remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)


def remove_punc(text: str) -> str:
    if not is_number(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    else:
        return text


def lower(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list:
    return re.split(" ", text)


def normalize_number(text: str) -> str:
    if is_number(text):
        return str(to_number(text))
    else:
        return text


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [white_space_fix(remove_articles(normalize_number(remove_punc(lower(token)))))
             for token in tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized


def build_tags(input_ids: list, offsets: list, answers: list,
    scheme: str, cls_id: int, sep_id: int, pad_id: int) -> list:

    tags = [0] * len(input_ids)

    for answer in answers:
        first = True

        for i, offset in enumerate(offsets):
            if answer[0] < offset[1] and offset[0] < answer[1] and \
                input_ids[i] not in (cls_id, sep_id, pad_id):

                if first and scheme == 'bio':
                    tags[i] = 2
                    first = False
                
                else:
                    tags[i] = 1
    
    return tags


def smooth_tags(tags: list, word_idx: list, scheme: str = 'io') -> list:
    for i in range(len(tags)):
        if word_idx[i] > 0:
            end = i + 1

            while end < len(tags) and word_idx[end] == word_idx[i]:
                end += 1
                
            if sum(tags[i:end]) > 0:
                tags[i] = 2 if scheme == 'bio' else 1

                for j in range(i + 1, end):
                    tags[j] = 1
    
    return tags


def is_cell_in_answers(mapping: dict, row_i: int, cell_i: int) -> bool:
    if mapping is not None and 'table' in mapping:
        for loc in mapping['table']:
            if loc[0] == row_i and loc[1] == cell_i:
                return True
        
    return False


def rank_paragraphs(question: str, paragraphs: list) -> list:
    order, corpus = [], [question]

    for paragraph in paragraphs:
        corpus.append(paragraph['text'])
        order.append(paragraph['order'] - 1)
        
    tfidf = TfidfVectorizer().fit_transform(corpus)
    cos = linear_kernel(tfidf[0:1], tfidf).flatten()[1:]
    sim = sorted(enumerate(cos), key=lambda x:x[1])
    idx = [i[0] for i in sim][::-1]
    return [order[i] for i in idx]


def extract_words(text: str) -> tuple:
    words, offsets = [], []
    prev_is_whitespace = True

    for i, c in enumerate(text):
        if c in (' ', ' ', '\t', '\r', '\n') or ord(c) == 0x202F:
            prev_is_whitespace = True
        
        elif c == '?':
            words.append(c)
            offsets.append([i, i + 1])
            prev_is_whitespace = False

        else:
            if prev_is_whitespace:
                words.append(c)
                offsets.append([i, i + 1])
                
            else:
                words[-1] += c
                offsets[-1][1] = i + 1
                
            prev_is_whitespace = False
        
    offsets[-1][1] = i + 1
    return words, [tuple(o) for o in offsets]


def extract_nums(words: list) -> list:
    nums = []

    for i in range(len(words)):
        if to_number(words[i]) is not None:
            num = to_number(words[i])

            if i > 0 and num >= 1 and num <= 31 \
                and any(words[i - 1].lower() == month for month in MONTHS):
                nums.append(float('nan'))
                
            elif any(str(year) in words[i] for year in range(1900, 2100)):
                nums.append(float('nan'))
                
            else:
                nums.append(num)
            
        else:
            nums.append(float('nan'))
        
    return nums


def extract_paragraph_answers(mapping: dict, paragraph: dict, text: str) -> list:
    answers = []
    
    if mapping is not None and 'paragraph' in mapping:
        if str(paragraph['order']) in mapping['paragraph']:
            for span in mapping['paragraph'][str(paragraph['order'])]:
                answers.append((len(text) + span[0], len(text) + span[1]))
    
    return answers


def replace_nums(input_ids: list, offsets: list, nums: list, 
    word_idx: list, num_id: int) -> tuple:

    o_input_ids, o_offsets, o_word_idx = [], [], []
    i = 0

    while i < len(input_ids):
        o_input_ids.append(input_ids[i])
        o_offsets.append(offsets[i])
        o_word_idx.append(word_idx[i])

        if not math.isnan(nums[word_idx[i]]):
            o_input_ids[-1] = num_id
            j = i + 1

            while j < len(input_ids) and word_idx[i] == word_idx[j]:
                o_offsets[-1] = (o_offsets[-1][0], offsets[j][1])
                j += 1
                
            i = j

        else:
            i += 1
        
    return o_input_ids, o_offsets, o_word_idx


def map_word_idx(token_offsets: list, word_offsets: list) -> list:
    idx = [0] * len(token_offsets)

    for t, t_offset in enumerate(token_offsets):
        for w, w_offset in enumerate(word_offsets):
            if (w_offset[0] <= t_offset[0] and t_offset[1] <= w_offset[1]) or \
                (t_offset[0] <= w_offset[0] and t_offset[1] > w_offset[0]) or \
                (t_offset[1] >= w_offset[1] and t_offset[0] < w_offset[1]):
                idx[t] = w + 1

    return idx


def adjust_offsets(offsets: list, text: str):
    return [(o[0] + len(text), o[1] + len(text)) for o in offsets]


def get_opr_name(opr_code: int):
    for key, code in OPERATORS.items():
        if code == opr_code:
            return key


def get_opr_map(oprs: list):
    opr_map = []

    for opr in oprs:
        opr_map.append(OPERATORS[opr])
    
    return opr_map


def get_order(sample: dict) -> int:
    if 'operation_order' in sample['question'] and \
        sample['question']['operation_order'] == 'reverse':
        return 1
        
    return 0


def get_scale(sample: dict) -> int:
    if 'scale' in sample['question']:
        return SCALES.index(sample['question']['scale'])
        
    return 0


def build_tensor(inputs: list, stype: str, key: str, dtype=torch.long):
    data = [getattr(n, stype)[key] for n in inputs]
    return torch.tensor(data, dtype=dtype)
