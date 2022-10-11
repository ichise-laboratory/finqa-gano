import logging
from collections import namedtuple
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from gano.finqa.tatqa.datasets import TatQADatasetForLM
from gano.finqa.tatqa.tagop.configs import (
    OPERATORS, OPR_NON_ORDER_CLASSES, OPR_ORDER_CLASSES)
from gano.finqa.tatqa.tagop.reasoning import adjust_numbers
from gano.finqa.tatqa.tagop.utils import (adjust_offsets, build_tags, build_tensor,
    extract_paragraph_answers, extract_nums, extract_words, 
    get_order, get_opr_map, get_scale, is_cell_in_answers, 
    map_word_idx, rank_paragraphs, smooth_tags, to_number)


class TagOpDataset(TatQADatasetForLM):
    empty_token = '<empty>'

    def __init__(
        self,
        *args,
        fill_empty_cells: bool = False,
        max_token_len: int = 512,
        n_paragraphs: int = None,
        scheme: str = 'io',
        **kwargs):

        super().__init__(*args, **kwargs)

        self._unknown_opr = len(OPERATORS) + 1
        self.fill_empty_cells = fill_empty_cells
        self.n_paragraphs = n_paragraphs
        self.num_labels = len(scheme)
        self.scheme = scheme

        self.opr_order_cls = get_opr_map(OPR_ORDER_CLASSES)
        self.opr_non_order_cls = get_opr_map(OPR_NON_ORDER_CLASSES)
        self.opr_num_cls = self.opr_order_cls + self.opr_non_order_cls

        if self.model_name in self.tokenizer.max_model_input_sizes:
            self.max_token_len = self.tokenizer.max_model_input_sizes[self.model_name]
        else:
            self.max_token_len = max_token_len

        if fill_empty_cells:
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': [self.empty_token]})
            self.empty_id = self.tokenizer.get_added_vocab()[self.empty_token]

    def process(self, split: str, samples: list) -> Tuple[TensorDataset, list]:
        texts, offsets, inputs = {}, {}, []
        logging.info(f'Tokenizing and processing samples [{split}].')

        for i, sample in enumerate(tqdm(samples)):
            n = self._process_sample(i, sample)
            texts[str(i)] = n.s['text']
            offsets[str(i)] = n.s['offsets']
            inputs.append(n)

        ids_map = [n.s['qid'] for n in inputs]

        if split != 'predict':
            inputs = self._process_filter(inputs)

        inputs = self._process_padding(inputs)

        self.texts[split] = texts
        self.offsets[split] = offsets

        status = f'Data statistics [{split}]:\n'
        status += f'.. All: {len(texts)}\n'
        status += f'.. Filtered: {len(inputs)}'
        logging.info(status)

        return self._process_tensors(inputs, samples), ids_map

    def _process_sample(self, idx: int, sample: dict) -> tuple:
        question = sample['question']
        table = sample['table']
        paragraphs = sample['paragraphs']

        if 'mapping' in sample['question']:
            mapping = sample['question']['mapping']
        else:
            mapping = None

        n = namedtuple('sample', ['s', 'q' 't', 'p'])
        n.s = {'qid': question['uid']}
        n.q = self._process_question(question, mapping)
        n.t = self._process_table(table, mapping)
        n.p = self._process_paragraphs(question, paragraphs, mapping)
        
        n.s['input_ids'] = n.q['input_ids'] + n.t['input_ids'] + n.p['input_ids']
        n.s['offsets'] = n.q['offsets'] + adjust_offsets(n.t['offsets'], n.q['text']) + \
            adjust_offsets(n.p['offsets'], n.q['text'] + n.t['text'])

        n.s['text'] = n.q['text'] + n.t['text'] + n.p['text']
        n.s['tags'] = n.q['tags'] + n.t['tags'] + n.p['tags']
        n.s['types'] = [1] * len(n.q['tags']) + [2] * len(n.t['tags']) + [3] * len(n.p['tags'])

        n.t['mask'] = [0] * len(n.q['input_ids']) + [1] * len(n.t['input_ids']) + [0] * len(n.p['input_ids'])
        n.p['mask'] = [0] * len(n.q['input_ids']) + [0] * len(n.t['input_ids']) + [1] * len(n.p['input_ids'])

        n.t['cell_idx'] = [0] * len(n.q['input_ids']) + n.t['cell_idx'] + [0] * len(n.p['input_ids'])
        n.p['word_idx'] = [0] * len(n.q['input_ids']) + [0] * len(n.t['input_ids']) + n.p['word_idx']
        
        n.t['cell_pos'] = [None] * len(n.q['input_ids']) + n.t['cell_pos'] + [None] * len(n.p['input_ids'])

        limit = self.max_token_len - 1

        if len(n.s['input_ids']) >= self.max_token_len:
            n.s['input_ids'] = n.s['input_ids'][:limit] + [self.sep_id]
            n.s['offsets'] = n.s['offsets'][:limit] + [(0, 0)]
            n.s['tags'] = n.s['tags'][:limit] + [0]
            n.s['types'] = n.s['types'][:limit] + [3]

            n.t['mask'] = n.t['mask'][:limit] + [0]
            n.t['cell_idx'] = n.t['cell_idx'][:limit] + [0]
            n.t['cell_pos'] = n.t['cell_pos'][:limit] + [None]

            n.p['mask'] = n.p['mask'][:limit] + [0]
            n.p['word_idx'] = n.p['word_idx'][:limit] + [0]
        
        n.s['idx'] = idx
        n.s['order_tag'] = get_order(sample)
        n.s['scale_tag'] = get_scale(sample)
        n.s['lens'] = [0, 0, 0]

        for t in n.s['types']:
            n.s['lens'][t - 1] += 1

        if 'operator' not in question or question['operator'] is None:
            n.s['opr_tag'] = self._unknown_opr
        else:
            n.s['opr_tag'] = OPERATORS[question['operator']]

        return self._process_derivation(sample, n)
    
    def _process_question(self, question: dict, mapping: dict, ignore_answers: bool = True):
        text = question['question']
        tokenized = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = tokenized['input_ids']
        offsets = tokenized['offset_mapping']

        words, word_offsets = extract_words(text)
        nums = extract_nums(words)
        word_idx = map_word_idx(offsets, word_offsets)

        if not ignore_answers and mapping is not None and 'question' in mapping:
            answers = mapping['question']  
        else:
            answers = []

        tags = build_tags(input_ids, offsets, answers, self.scheme, 
            self.cls_id, self.sep_id, self.pad_id)
        
        tags = smooth_tags(tags, word_idx, self.scheme)
        word_idx[0] = word_idx[-1] = 0

        return {'text': text, 'input_ids': input_ids, 'offsets': offsets, 
            'tags': tags, 'nums': nums, 'word_idx': word_idx}
    
    def _process_table(self, table: dict, mapping: dict):
        input_ids, offsets, tags, nums, cell_idx, cell_pos = [], [], [], [], [], []
        text, counter = '', 1

        for row_i, row in enumerate(table['table']):
            for cell_i, cell in enumerate(row):
                if len(cell['text']):
                    tokenized = self.tokenizer(cell['text'], return_offsets_mapping=True)
                    cell_offsets = tokenized['offset_mapping'][1:-1]
                    input_ids += tokenized['input_ids'][1:-1]

                    offsets += [(o[0] + len(text), o[1] + len(text)) for o in cell_offsets]
                    cell_idx += [counter] * len(cell_offsets)
                    cell_pos += [(row_i, cell_i) for _ in range(len(cell_offsets))]

                    if is_cell_in_answers(mapping, row_i, cell_i):
                        if self.scheme == 'bio':
                            tags += [2] + [1] * (len(cell_offsets) - 1)
                        
                        else:
                            tags += [1] * len(cell_offsets)
                    
                    else:
                        tags += [0] * len(cell_offsets)

                    nums.append(to_number(cell['text']) or float('nan'))

                    text += cell['text']
                    counter += 1
                
                elif self.fill_empty_cells:
                    input_ids.append(self.empty_id)
                    offsets.append((0, 0))
                    cell_idx.append(counter)
                    cell_pos.append((row_i, cell_i))
                    tags.append(0)
                    nums.append(float('nan'))
                    counter += 1
        
        input_ids.append(self.sep_id)
        offsets.append((0, 0))
        tags.append(0)
        cell_idx.append(0)
        cell_pos.append(None)

        return {'text': text, 'input_ids': input_ids, 'offsets': offsets, \
            'tags': tags, 'nums': nums, 'cell_idx': cell_idx, 'cell_pos': cell_pos}
    
    def _process_paragraphs(self, question: dict, paragraphs: list, mapping: dict):
        text, answers = '', []
        order = rank_paragraphs(question['question'], paragraphs)
        paragraphs = [paragraphs[i] for i in order]

        if self.n_paragraphs is not None:
            paragraphs = paragraphs[:self.n_paragraphs]

        for paragraph in paragraphs:
            answers += extract_paragraph_answers(mapping, paragraph, text)
            text += paragraph['text'] + ' ' if paragraph['text'][-1] != ' ' else paragraph['text']
        
        tokenized = self.tokenizer(text, truncation=True, return_offsets_mapping=True)
        input_ids = tokenized['input_ids'][1:]
        offsets = tokenized['offset_mapping'][1:]

        words, word_offsets = extract_words(text)
        nums = extract_nums(words)
        word_idx = map_word_idx(offsets, word_offsets)

        tags = build_tags(input_ids, offsets, answers, self.scheme,
            self.cls_id, self.sep_id, self.pad_id)

        tags = smooth_tags(tags, word_idx, self.scheme)

        return {'text': text, 'input_ids': input_ids, 'offsets': offsets, \
            'tags': tags, 'nums': nums, 'word_idx': word_idx}
    
    def _process_filter(self, inputs: list) -> list:
        filtered = []

        for n in inputs:
            include = True

            if sum(n.s['tags']) == 0:
                include = False
                
            if n.s['opr_tag'] == self._unknown_opr:
                include = False
            
            if n.s['opr_tag'] in self.opr_num_cls and n.d['order'] is None:
                include = False
            
            if include:
                filtered.append(n)

        return filtered
    
    def _process_derivation(self, sample: dict, n: tuple):
        setattr(n, 'd', {})
        n.d['pos'] = []
        n.d['nums'] = []
        n.d['order'] = None

        if n.s['opr_tag'] in self.opr_num_cls:
            answer = to_number(str(sample['question']['answer']), number=True)
            opr = sample['question']['operator']
            scale = sample['question']['scale']
            cell_idx, word_idx = None, None
            nums, pos = [], []

            for i in range(len(n.s['input_ids'])):
                if n.s['tags'][i] != 0:
                    if n.t['cell_idx'][i] > 0 and \
                        not np.isnan(n.t['nums'][n.t['cell_idx'][i] - 1]):
                        if cell_idx == n.t['cell_idx'][i]:
                            pos[-1].append(i)
                        
                        else:
                            assert n.t['cell_idx'] != n.t['cell_idx'][i - 1]
                            cell_idx = n.t['cell_idx'][i]
                            nums.append(n.t['nums'][cell_idx - 1])
                            pos.append([i])
                    
                    if n.p['word_idx'][i] > 0 and \
                        not np.isnan(n.p['nums'][n.p['word_idx'][i] - 1]):
                        if word_idx == n.p['word_idx'][i]:
                            pos[-1].append(i)
                        
                        else:
                            assert n.p['word_idx'] != n.p['word_idx'][i - 1]
                            word_idx = n.p['word_idx'][i]
                            nums.append(n.p['nums'][word_idx - 1])
                            pos.append([i])
            
            for i in range(len(pos)):
                pos[i] = (min(pos[i]), max(pos[i]) + 1)

            n.d['pos'] = pos
            n.d['nums'] = nums
            n.d['order'] = adjust_numbers(nums, answer, opr, scale)

        return n

    def _process_padding(self, inputs: list) -> list:
        max_s_len = max(len(n.s['input_ids']) for n in inputs)
        max_t_num_len = max(len(n.t['nums']) for n in inputs)
        max_p_num_len = max(len(n.p['nums']) for n in inputs)

        for n in inputs:
            s_pad = max_s_len - len(n.s['input_ids'])
            n.s['attention_mask'] = [1] * len(n.s['input_ids']) + [0] * s_pad
            n.s['input_ids'] += [self.pad_id] * s_pad
            n.s['tags'] += [0] * s_pad

            n.t['mask'] += [0] * s_pad
            n.t['cell_idx'] += [0] * s_pad
            n.p['mask'] += [0] * s_pad
            n.p['word_idx'] += [0] * s_pad

            t_pad = max_t_num_len - len(n.t['nums'])
            n.t['nums'] += [0] * t_pad

            p_pad = max_p_num_len - len(n.p['nums'])
            n.p['nums'] += [0] * p_pad
        
        return inputs
    
    def _process_tensors(self, inputs: list, samples: dict) -> TensorDataset:
        sample_idx = build_tensor(inputs, 's', 'idx')
        input_ids = build_tensor(inputs, 's', 'input_ids')
        attention_masks = build_tensor(inputs, 's', 'attention_mask')
        input_tags = build_tensor(inputs, 's', 'tags')
        opr_tags = build_tensor(inputs, 's', 'opr_tag')
        order_tags = build_tensor(inputs, 's', 'order_tag')
        scale_tags = build_tensor(inputs, 's', 'scale_tag')
        type_lens = build_tensor(inputs, 's', 'lens')

        tbl_masks = build_tensor(inputs, 't', 'mask')
        tbl_nums = build_tensor(inputs, 't', 'nums', dtype=torch.float)
        tbl_cell_idx = build_tensor(inputs, 't', 'cell_idx')
        
        prg_masks = build_tensor(inputs, 'p', 'mask')
        prg_nums = build_tensor(inputs, 'p', 'nums', dtype=torch.float)
        prg_word_idx = build_tensor(inputs, 'p', 'word_idx')

        return TensorDataset(sample_idx,
            tbl_masks, tbl_nums, tbl_cell_idx,
            prg_masks, prg_nums, prg_word_idx,
            input_tags, opr_tags, order_tags, 
            scale_tags, type_lens,
            input_ids, attention_masks)

    def _process_types(self, inputs: list) -> list:
        type_lens = []

        for n in inputs:
            lens = [0, 0, 0]

            for t in n.s['types']:
                lens[t - 1] += 1
            
            type_lens.append(lens)
        
        return type_lens


class TagOpFinBertDataset(TagOpDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_token_len = 512
