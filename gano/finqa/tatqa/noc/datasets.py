import numpy as np
import torch
from torch.utils.data import TensorDataset

from gano.finqa.tatqa.tagop.datasets import TagOpDataset
from gano.finqa.tatqa.tagop.utils import build_tensor


class NocDataset(TagOpDataset):
    def _process_nums(self, n: tuple) -> tuple:
        tmap = [[-1, -1] for _ in range(len(n.t['nums']))]
        pmap = [[-1, -1] for _ in range(len(n.p['nums']))]

        for i in range(len(n.s['tags'])):
            cell_idx = n.t['cell_idx'][i] - 1
            word_idx = n.p['word_idx'][i] - 1

            if cell_idx >= 0 and not np.isnan(n.t['nums'][cell_idx]):
                if tmap[cell_idx][0] == -1:
                    tmap[cell_idx][0] = i
                
                tmap[cell_idx][1] = i + 1
            
            if word_idx >= 0 and not np.isnan(n.p['nums'][word_idx]):
                if pmap[word_idx][0] == -1:
                    pmap[word_idx][0] = i
                
                pmap[word_idx][1] = i + 1
        
        n.t['nums_map'] = tmap
        n.p['nums_map'] = pmap
        return n

    def _process_order(self, n: tuple) -> tuple:
        setattr(n, 'o', {})
        n.o['tags'] = [0] * len(n.s['tags'])
        n.o['mask'] = [0] * len(n.s['tags'])
        n.o['nums'] = [-float('inf'), -float('inf')]

        if n.s['opr_tag'] in self.opr_order_cls and n.d['order'] is not None:
            pos = n.d['pos']
            nums = n.d['nums']
            order = n.d['order']

            if order == 1:
                for p in range(pos[0][0], pos[0][1]): n.o['tags'][p] = 1
                for p in range(pos[1][0], pos[1][1]): n.o['tags'][p] = 2
                n.o['nums'] = [nums[0], nums[1]]
                    
            elif order == -1:
                for p in range(pos[0][0], pos[0][1]): n.o['tags'][p] = 2
                for p in range(pos[1][0], pos[1][1]): n.o['tags'][p] = 1
                n.o['nums'] = [nums[1], nums[0]]
                    
            for p in pos[0]:
                n.o['mask'][p] = 1

            for p in pos[1]: 
                n.o['mask'][p] = 1
        
        else:
            n.o['nums'] = [float('nan'), float('nan')]

        return n
    
    def _process_padding(self, inputs: list) -> list:
        inputs = super()._process_padding(inputs)
        max_s_len = max(len(n.s['input_ids']) for n in inputs)
        max_d_num_len = max(len(n.d['nums']) for n in inputs)
        max_t_num_len = max(len(n.t['nums']) for n in inputs)
        max_p_num_len = max(len(n.p['nums']) for n in inputs)

        for n in inputs:
            d_pad = max_d_num_len - len(n.d['nums'])
            n.d['nums'] += [float('nan')] * d_pad
            n.d['pos'] += [[-1, -1] for _ in range(d_pad)]

            s_pad = max_s_len - len(n.o['tags'])
            n.o['tags'] += [0] * s_pad
            n.o['mask'] += [0] * s_pad

            t_pad = max_t_num_len - len(n.t['nums_map'])
            n.t['nums_map'] += [[-1, -1] for _ in range(t_pad)]

            p_pad = max_p_num_len - len(n.p['nums_map'])
            n.p['nums_map'] += [[-1, -1] for _ in range(p_pad)]

        return inputs

    def _process_sample(self, idx: int, sample: dict) -> tuple:
        n = super()._process_sample(idx, sample)
        n = self._process_nums(n)
        return self._process_order(n)
    
    def _process_tensors(self, inputs: list, samples: dict) -> TensorDataset:
        sample_idx = build_tensor(inputs, 's', 'idx')
        input_ids = build_tensor(inputs, 's', 'input_ids')
        attention_masks = build_tensor(inputs, 's', 'attention_mask')
        input_tags = build_tensor(inputs, 's', 'tags')
        opr_tags = build_tensor(inputs, 's', 'opr_tag')
        scale_tags = build_tensor(inputs, 's', 'scale_tag')
        type_lens = build_tensor(inputs, 's', 'lens')

        tbl_masks = build_tensor(inputs, 't', 'mask')
        tbl_nums = build_tensor(inputs, 't', 'nums', dtype=torch.float)
        tbl_cell_idx = build_tensor(inputs, 't', 'cell_idx')
        tbl_nums_map = build_tensor(inputs, 't', 'nums_map')
        
        prg_masks = build_tensor(inputs, 'p', 'mask')
        prg_nums = build_tensor(inputs, 'p', 'nums', dtype=torch.float)
        prg_word_idx = build_tensor(inputs, 'p', 'word_idx')
        prg_nums_map = build_tensor(inputs, 'p', 'nums_map')

        order_tags = build_tensor(inputs, 'o', 'tags')
        order_nums = build_tensor(inputs, 'o', 'nums', dtype=torch.float)
        order_masks = build_tensor(inputs, 'o', 'mask')

        deriv_nums = build_tensor(inputs, 'd', 'nums', dtype=torch.float)
        deriv_pos = build_tensor(inputs, 'd', 'pos')

        return TensorDataset(sample_idx,
            tbl_masks, tbl_nums, tbl_cell_idx, tbl_nums_map,
            prg_masks, prg_nums, prg_word_idx, prg_nums_map,
            input_tags, opr_tags, order_tags, order_nums, order_masks,
            scale_tags, type_lens, deriv_nums, deriv_pos,
            input_ids, attention_masks)
