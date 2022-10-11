import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from allennlp.nn import util
from transformers.modeling_utils import PreTrainedModel

from gano.finqa.tatqa.metrics import get_metrics
from gano.finqa.tatqa.models import TatQALightningForLM
from gano.finqa.tatqa.tagop.configs import (
    OPR_CLASSES, SCALES, OPERATORS, OPR_ORDER_CLASSES, 
    OPR_NON_ORDER_CLASSES, OPR_SPAN_CLASSES)
from gano.finqa.tatqa.tagop.nn import FFNLayer, reduce_max, reduce_mean
from gano.finqa.tatqa.tagop.utils import (
    get_tbl_spans, get_prg_spans, get_nums, get_opr_map)


class TagOpModel(nn.Module):
    def __init__(
        self, 
        encoder: PreTrainedModel, 
        config: dict, 
        num_labels: int = 2, 
        dropout_prob: float = None,
        use_lstm: bool = False):

        super().__init__()

        self.encoder = encoder
        self.config = config

        hidden_size = self.config.hidden_size
        dropout_prob = dropout_prob or getattr(self.config, 'hidden_dropout_prob', 0)

        self.opr_classifier = FFNLayer(hidden_size, hidden_size, len(OPERATORS), dropout_prob)
        self.scale_classifier = FFNLayer(3 * hidden_size, hidden_size, len(SCALES), dropout_prob)
        self.tag_classifier = FFNLayer(hidden_size, hidden_size, num_labels, dropout_prob)
        self.order_classifier = FFNLayer(hidden_size, hidden_size, 2, dropout_prob)

        self.opr_loss_fn = nn.CrossEntropyLoss()
        self.scale_loss_fn = nn.CrossEntropyLoss()
        self.NLLLoss = nn.NLLLoss(reduction='sum')
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        self.opr_order_cls = get_opr_map(OPR_ORDER_CLASSES)

        if use_lstm:
            logging.info('Using LSTM layer.')
            self.lstm = nn.LSTM(
                hidden_size, 
                hidden_size // 2, 
                num_layers=2,
                dropout=dropout_prob,
                bidirectional=True)
        
        self.use_lstm = use_lstm
    
    def forward(
        self,
        model_input: dict,
        tbl_masks: torch.LongTensor,
        tbl_nums: torch.FloatTensor,
        tbl_cell_idx: torch.LongTensor,
        prg_masks: torch.LongTensor,
        prg_nums: torch.FloatTensor,
        prg_word_idx: torch.LongTensor,
        input_tags: torch.LongTensor = None,
        opr_tags: torch.LongTensor = None,
        order_tags: torch.LongTensor = None) -> tuple:

        outputs = self.encoder(**model_input)
        seq_emb = outputs[0]
        batch_size = seq_emb.shape[0]
        device = tbl_masks.device

        cls_emb = seq_emb[:, 0, :]

        tbl_emb = util.replace_masked_values(seq_emb, tbl_masks.unsqueeze(-1).bool(), 0)
        tbl_logits = self.tag_classifier(tbl_emb)
        tbl_logits = util.masked_log_softmax(tbl_logits, mask=None)
        tbl_logits = util.replace_masked_values(tbl_logits, tbl_masks.unsqueeze(-1).bool(), 0)

        prg_emb = util.replace_masked_values(seq_emb, prg_masks.unsqueeze(-1).bool(), 0)

        if self.use_lstm:
            prg_emb, _ = self.lstm(prg_emb)

        prg_logits = self.tag_classifier(prg_emb)
        prg_logits = util.masked_log_softmax(prg_logits, mask=None)
        prg_logits = util.replace_masked_values(prg_logits, prg_masks.unsqueeze(-1).bool(), 0)

        if input_tags is not None:
            tbl_tags = util.replace_masked_values(input_tags.float(), tbl_masks.bool(), 0)
            prg_tags = util.replace_masked_values(input_tags.float(), prg_masks.bool(), 0)
        
        else:
            tbl_tags = prg_tags = None

        prg_reduce_mean = torch.mean(prg_emb, dim=1)
        tbl_reduce_mean = torch.mean(tbl_emb, dim=1)
        cls_tbl_prg_emb = torch.cat((cls_emb, tbl_reduce_mean, prg_reduce_mean), dim=-1)

        tbl_max_logits = reduce_max(tbl_logits[:, :, 1], tbl_cell_idx)
        tbl_mean_emb = reduce_mean(tbl_emb, tbl_cell_idx)
        prg_max_logits = reduce_max(prg_logits[:, :, 1], prg_word_idx)
        prg_mean_emb = reduce_mean(prg_emb, prg_word_idx)

        tbl_num_masks = reduce_mean(tbl_masks, tbl_cell_idx)
        prg_num_masks = reduce_mean(prg_masks, prg_word_idx)

        tbl_num_logits = util.replace_masked_values(tbl_max_logits, tbl_num_masks.bool(), -1e+5)
        prg_num_logits = util.replace_masked_values(prg_max_logits, prg_num_masks.bool(), -1e+5)

        srt_tbl_logits, srt_tbl_idx = torch.sort(tbl_num_logits, dim=-1, descending=True)
        srt_prg_logits, srt_prg_idx = torch.sort(prg_num_logits, dim=-1, descending=True)

        srt_tbl_logits, srt_tbl_idx = srt_tbl_logits[:, :2], srt_tbl_idx[:, :2]
        srt_prg_logits, srt_prg_idx = srt_prg_logits[:, :2], srt_prg_idx[:, :2]

        seq_logits = torch.cat((srt_prg_logits, srt_tbl_logits), dim=1)
        opr_logits = self.opr_classifier(cls_emb)
        scale_logits = self.scale_classifier(cls_tbl_prg_emb)

        _, srt_seq_idx = torch.sort(seq_logits, dim=-1, descending=True)
        opr_cls = torch.argmax(opr_logits, dim=-1)

        top2_nums = torch.zeros(batch_size, 2, device=device)
        top2_order_tags = torch.zeros(batch_size, device=device)
        top2_emb = torch.zeros(batch_size, 2, self.hidden_size, device=device)
        top2_emb_bw = torch.zeros(batch_size, 2, self.hidden_size, device=device)

        num_i = gold_i = 0

        for i in range(batch_size):
            if opr_cls[i] in self.opr_order_cls:
                srt_idx = srt_seq_idx[i]

                for j in (0, 1):
                    if srt_idx[j] > 1:
                        top2_nums[num_i, j] = tbl_nums[i][srt_tbl_idx[i, srt_idx[j] - 2] - 1]
                        top2_emb[num_i, j, :] = tbl_mean_emb[i, srt_tbl_idx[i, srt_idx[j] - 2], :]
                    
                    else:
                        top2_nums[num_i, j] = prg_nums[i][srt_prg_idx[i, srt_idx[j]] - 1]
                        top2_emb[num_i, j, :] = prg_mean_emb[i, srt_prg_idx[i, srt_idx[j]], :]

                num_i += 1

                if opr_tags is None or opr_tags[i] not in self.opr_order_cls:
                    continue
            
                top2_order_tags[gold_i] = order_tags[i]

                for j in (0, 1):
                    if srt_idx[j] > 1:
                        top2_emb_bw[gold_i, j, :] = tbl_mean_emb[i, srt_tbl_idx[i, srt_idx[j] - 2], :]
                    
                    else:
                        top2_emb_bw[gold_i, j, :] = prg_mean_emb[i, srt_prg_idx[i, srt_idx[j]], :]
                
                gold_i += 1

        top2_order_logits = self.order_classifier(torch.mean(top2_emb[:num_i], dim=1))
        top2_nums_pred = top2_nums[:num_i]
        top2_order_logits_bw = self.order_classifier(torch.mean(top2_emb_bw[:gold_i], dim=1))
        top2_order_tags = top2_order_tags[:gold_i]

        return tbl_logits, tbl_tags, prg_logits, prg_tags, opr_logits, scale_logits, \
            top2_nums_pred, top2_order_logits, top2_order_logits_bw, top2_order_tags, \
            num_i, gold_i, outputs
    
    def fit(
        self,
        tbl_logits: torch.FloatTensor,
        tbl_tags: torch.FloatTensor,
        prg_logits: torch.FloatTensor,
        prg_tags: torch.FloatTensor,
        opr_logits: torch.FloatTensor,
        opr_tags: torch.LongTensor,
        scale_logits: torch.FloatTensor,
        scale_tags: torch.LongTensor,
        top2_order_logits_bw: torch.FloatTensor,
        top2_order_tags: torch.FloatTensor,
        gold_i: int) -> tuple:

        tbl_logits = tbl_logits.transpose(1, 2)
        prg_logits = prg_logits.transpose(1, 2)

        tbl_loss = self.NLLLoss(tbl_logits, tbl_tags.long())
        prg_loss = self.NLLLoss(prg_logits, prg_tags.long())
        opr_loss = self.opr_loss_fn(opr_logits, opr_tags)
        scale_loss = self.scale_loss_fn(scale_logits, scale_tags)

        if gold_i > 0:
            top2_order_logits_bw = util.masked_log_softmax(top2_order_logits_bw, mask=None)
            order_loss = self.NLLLoss(top2_order_logits_bw, top2_order_tags.long())
        
        else:
            order_loss = torch.tensor(0, dtype=torch.float, device=top2_order_logits_bw.device)
        
        loss = tbl_loss + prg_loss + opr_loss + order_loss + scale_loss

        return loss, tbl_loss, prg_loss, opr_loss, \
            order_loss, scale_loss, top2_order_tags.shape[0]


class TagOpLightning(TatQALightningForLM):
    OPR_MAP = ('Span-in-text', 'Cell-in-table', 'Spans', 'Sum', 'Count', 'Average',
        'Multiplication', 'Division', 'Difference', 'Change ratio', 'ignore')
    METRIC_KEYS = ('opr', 'scale', 'em.ext.hyb', 'f1.ext.hyb', 
        'em.ext.tbl', 'f1.ext.tbl', 'em.ext.prg', 'f1.ext.prg')

    def __init__(
        self, 
        *args, 
        scheme: str = 'io', 
        dropout_prob: float = None, 
        use_lstm: bool = False,
        **kwargs):

        self.dropout_prob = dropout_prob
        self.num_labels = len(scheme)
        self.use_lstm = use_lstm

        self.opr_cls = get_opr_map(OPR_CLASSES)
        self.opr_order_cls = get_opr_map(OPR_ORDER_CLASSES)
        self.opr_non_order_cls = get_opr_map(OPR_NON_ORDER_CLASSES)
        self.opr_num_cls = get_opr_map(OPR_NON_ORDER_CLASSES + OPR_ORDER_CLASSES)
        self.opr_span_cls = get_opr_map(OPR_SPAN_CLASSES)

        super().__init__(*args, **kwargs)

    def init_model(self, model_params: dict = None) -> None:
        super().init_model(self.update_args(model_params, {
            'num_labels': self.num_labels,
            'dropout_prob': self.dropout_prob,
            'use_lstm': self.use_lstm}))

    def init_model_cls(self) -> None:
        super().init_model_cls()
        self.model_cls = TagOpModel
    
    def epoch_end(self, split: str) -> None:
        if split == 'predict':
            self.outputs[split] = {
                'preds': self.combine_outputs(self.outputs[split], 'preds'),
                'records': self.combine_outputs(self.outputs[split], 'records')}

        else:
            for split in self._get_splits(split):
                self.outputs[split] = {
                    'preds': self.combine_outputs(self.outputs[split], 'preds'),
                    'records': self.combine_outputs(self.outputs[split], 'records'),
                    'results': self.combine_outputs(self.outputs[split], 'results')}
                
                results = self.outputs[split]['results']
                records = self.metrics[split].get_raw()
                
                for record in records:
                    results[record['qid']] = {
                        **results[record['qid']],
                        'f1': record['f1'],
                        'em': record['em'],
                        'f1.span': record['span_f1'],
                        'em.span': record['span_em']}

                self.results[split].append({
                    **self._aggregate_results(split),
                    **self.compute_metric(self.metrics[split])})

    def predict(
        self,
        sample_idx: np.ndarray,
        tbl_pred: np.ndarray,
        tbl_nums: np.ndarray,
        tbl_cell_idx: np.ndarray,
        prg_pred: np.ndarray,
        prg_nums: np.ndarray,
        prg_word_idx: np.ndarray,
        opr_pred: np.ndarray,
        scale_pred: np.ndarray,
        top2_nums_pred: np.ndarray,
        top2_order_logits: np.ndarray,
        num_i: int,
        split: str) -> tuple:

        preds = {}
        answer_pred, answer_tags = [], []
        batch_size = tbl_pred.shape[0]

        texts = self.dataset.texts[split]
        offsets = self.dataset.offsets[split]

        top2_i = 0

        if num_i > 0:
            top2_order_pred = np.argmax(top2_order_logits, axis=1)
        
        for i in range(batch_size):
            si, dsi = sample_idx[i], str(sample_idx[i])
            tbl_pred_spans, tbl_pred_nums = None, None
            prg_pred_spans, prg_pred_nums = None, None
            answer, order_nums = '', None

            if opr_pred[i] in self.opr_span_cls:
                tbl_pred_spans = get_tbl_spans(tbl_pred[i], offsets[dsi], tbl_cell_idx[i], texts[dsi])
                prg_pred_spans = get_prg_spans(prg_pred[i], offsets[dsi], prg_word_idx[i], texts[dsi])
                answer = list(set(tbl_pred_spans + prg_pred_spans))

            elif opr_pred[i] == OPERATORS['count']:
                tbl_pred_spans = get_tbl_spans(tbl_pred[i], offsets[dsi], tbl_cell_idx[i], texts[dsi])
                prg_pred_spans = get_prg_spans(prg_pred[i], offsets[dsi], prg_word_idx[i], texts[dsi])
                answer = len(set(tbl_pred_spans + prg_pred_spans))

            elif opr_pred[i] in self.opr_non_order_cls:
                tbl_pred_nums, _ = get_nums(tbl_pred[i], tbl_nums[i])
                prg_pred_nums, _ = get_nums(prg_pred[i], prg_nums[i])
                nums = tbl_pred_nums + prg_pred_nums

                if len(nums) == 0:
                    answer = ''

                elif opr_pred[i] == OPERATORS['sum']:
                    answer = np.around(np.sum(nums), 4)

                elif opr_pred[i] == OPERATORS['multiply']:
                    answer = np.around(np.prod(nums), 4)
                    
                elif opr_pred[i] == OPERATORS['average']:
                    answer = np.around(np.mean(nums), 4)
                
                if answer != '':
                    if SCALES[int(scale_pred[i])] == 'percent':
                        answer *= 100

                    answer = np.around(answer, 2)
                
            elif top2_nums_pred.size > 0:
                order_pred = top2_order_pred[top2_i]
                num_a = top2_nums_pred[top2_i, order_pred]
                num_b = top2_nums_pred[top2_i, 1 - order_pred]
                order_nums = [float(num_a), float(num_b)]

                if not np.isnan(num_a) and not np.isnan(num_b):
                    if opr_pred[i] == OPERATORS['diff']:
                        answer = np.around(num_a - num_b, 4)
                            
                    elif opr_pred[i] == OPERATORS['divide'] and num_b != 0:
                        answer = np.around(num_a / num_b, 4)
                            
                    elif num_b != 0:
                        answer = np.around(num_a / num_b - 1, 4)
                                
                    if answer != '':
                        if SCALES[int(scale_pred[i])] == 'percent':
                            answer *= 100
                    
                        answer = np.around(answer, 2)

                top2_i += 1

            answer_pred, answer_tags, preds = self._predict_compile(
                si, answer, answer_pred, answer_tags,
                preds, opr_pred[i], scale_pred[i],
                tbl_pred_spans, tbl_pred_nums,
                prg_pred_spans, prg_pred_nums,
                order_nums, split)
        
        return answer_pred, answer_tags, preds
    
    def record(
        self,
        sample_idx: np.ndarray,
        tbl_logits: torch.FloatTensor,
        prg_logits: torch.FloatTensor,
        opr_logits: torch.FloatTensor,
        scale_logits: torch.FloatTensor,
        top2_nums_pred: np.ndarray,
        top2_order_logits: np.ndarray,
        split: str) -> dict:

        cpu = lambda x: x.detach().cpu().numpy()
        records = {}

        for i in range(sample_idx.shape[0]):
            qid = self.dataset.samples[split][sample_idx[i]]['question']['uid']
            records[qid] = {
                'tbl_logits': cpu(tbl_logits),
                'prg_logits': cpu(prg_logits),
                'opr_logits': cpu(opr_logits),
                'scale_logits': cpu(scale_logits),
                'top2_nums_pred': top2_nums_pred,
                'top2_order_logits': top2_order_logits}
        
        return records
    
    def evaluate(
        self,
        sample_idx: np.ndarray,
        tbl_pred: np.ndarray,
        tbl_tags: np.ndarray,
        tbl_cell_idx: np.ndarray,
        prg_pred: np.ndarray,
        prg_tags: np.ndarray,
        prg_word_idx: np.ndarray,
        answer_pred: list,
        answer_tags: list,
        opr_pred: np.ndarray,
        opr_tags: np.ndarray,
        scale_pred: np.ndarray,
        scale_tags: np.ndarray,
        split: str) -> dict:

        results = {}

        texts = self.dataset.texts[split]
        offsets = self.dataset.offsets[split]

        for i in range(len(answer_pred)):
            dsi = str(sample_idx[i])
            qid = self.dataset.ids_map[split][sample_idx[i]]
            result = {}

            s_tbl_pred = tbl_pred[i, :max(tbl_cell_idx[i]) + 1].tolist()
            s_prg_pred = prg_pred[i, :max(prg_word_idx[i]) + 1].tolist()

            s_tbl_pred = get_tbl_spans(s_tbl_pred, offsets[dsi], tbl_cell_idx[i], texts[dsi])
            s_prg_pred = get_prg_spans(s_prg_pred, offsets[dsi], prg_word_idx[i], texts[dsi])

            s_tbl_tags = tbl_tags[i, :max(tbl_cell_idx[i]) + 1].tolist()
            s_prg_tags = prg_tags[i, :max(prg_word_idx[i]) + 1].tolist()

            s_tbl_tags = get_tbl_spans(s_tbl_tags, offsets[dsi], tbl_cell_idx[i], texts[dsi])
            s_prg_tags = get_prg_spans(s_prg_tags, offsets[dsi], prg_word_idx[i], texts[dsi])

            s_opr_pred = opr_pred[i]
            s_scale_pred = scale_pred[i]

            result['opr'] = int(s_opr_pred == opr_tags[i])
            result['scale'] = int(s_scale_pred == scale_tags[i])

            self.metrics[split](
                {**answer_tags[i], 'qid': qid}, 
                answer_pred[i], 
                SCALES[int(scale_pred[i])],
                pred_op=self.OPR_MAP[opr_pred[i]], 
                gold_op=self.OPR_MAP[opr_tags[i]])

            if len(s_tbl_tags) + len(s_prg_tags):
                em, f1 = get_metrics(s_tbl_pred + s_prg_pred, s_tbl_tags + s_prg_tags)
                result['em.ext.hyb'] = em
                result['f1.ext.hyb'] = f1
            
            if len(s_tbl_tags):
                tbl_em, tbl_f1 = get_metrics(s_tbl_pred, s_tbl_tags)
                result['em.ext.tbl'] = tbl_em
                result['f1.ext.tbl'] = tbl_f1
            
            if len(s_prg_tags):
                prg_em, prg_f1 = get_metrics(s_prg_pred, s_prg_tags)
                result['em.ext.prg'] = prg_em
                result['f1.ext.prg'] = prg_f1
            
            assert len(s_tbl_tags) > 0 or len(s_prg_tags) > 0
            results[qid] = result
        
        return results
    
    def predict_step(self, batch: tuple, batch_idx: int):
        return self._step(batch, batch_idx, 'predict')

    def train_test_step(self, batch: tuple, batch_idx: int, split: str):
        return self._step(batch, batch_idx, split)
    
    def _aggregate_results(self, split: str) -> dict:
        metric_sums = {k: 0 for k in self.METRIC_KEYS}
        metric_sizes = {k: 0 for k in self.METRIC_KEYS}
        metrics = {}

        for result in self.outputs[split]['results'].values():
            for key in self.METRIC_KEYS:
                if key in result:
                    metric_sums[key] += result[key]
                    metric_sizes[key] += 1
        
        for key in self.METRIC_KEYS:
            if metric_sizes[key] > 0:
                metrics[key] = metric_sums[key] / metric_sizes[key]
            else:
                metrics[key] = 0

        return metrics

    def _predict_compile(
        self,
        sample_id: str,
        answer: Union[str, int, np.ndarray, list],
        answer_pred: list,
        answer_tags: list,
        preds: dict,
        opr_pred: np.ndarray,
        scale_pred: np.ndarray,
        tbl_pred_spans: list,
        tbl_pred_nums: list,
        prg_pred_spans: list,
        prg_pred_nums: list,
        order_nums: list,
        split: str):

        if isinstance(answer, list):
            if len(answer) == 0:
                answer = ''
                    
            elif len(answer) == 1:
                answer = answer[0]
            
        opr_map = {i: s for s, i in OPERATORS.items()}
        question = self.dataset.samples[split][sample_id]['question']
        answer_pred.append(answer)

        if split != 'predict':
            answer_tags.append({
                'answer': question['answer'],
                'answer_type': question['answer_type'],
                'answer_from': question['answer_from'],
                'scale': question['scale']})
            
        if isinstance(answer, list) or isinstance(answer, str):
            answer_str = answer
            
        else:
            answer_str = str(answer)
            
        if tbl_pred_nums is not None:
            tbl_pred_nums = [float(n) for n in tbl_pred_nums]
            
        if prg_pred_nums is not None:
            prg_pred_nums = [float(n) for n in prg_pred_nums]

        if split == 'predict':
            preds[question['uid']] = [answer_str, SCALES[int(scale_pred)]]
        
        else:
            preds[question['uid']] = {
                'answer': answer_str,
                'operator': opr_map[int(opr_pred)],
                'scale': SCALES[int(scale_pred)],
                'tbl_spans': tbl_pred_spans,
                'prg_spans': prg_pred_spans,
                'tbl_nums': tbl_pred_nums,
                'prg_nums': prg_pred_nums,
                'order_nums': order_nums}

        return answer_pred, answer_tags, preds
    
    def _predict_logits(self, logits: torch.FloatTensor) -> np.ndarray:
        cpu = lambda x: x.detach().cpu().numpy()
        return cpu(torch.argmax(logits, dim=-1))
    
    def _predict_seq(
        self, 
        logits: torch.FloatTensor, 
        word_idx: torch.LongTensor,
        type_lens: torch.LongTensor = None) -> torch.FloatTensor:
        pred = torch.argmax(logits, dim=-1).float()
        return reduce_max(pred, word_idx)

    def _prepare_model_input(self, tensors: tuple) -> dict:
        return {'input_ids': tensors[0], 'attention_mask': tensors[1]}
    
    def _step(self, batch: tuple, batch_idx: int, split: str) -> dict:
        cpu = lambda x: x.detach().cpu().numpy()

        sample_idx, tbl_masks, tbl_nums, tbl_cell_idx, \
            prg_masks, prg_nums, prg_word_idx, \
            input_tags, opr_tags, order_tags, \
            scale_tags, type_lens = batch[:12]
        
        tbl_logits, tbl_tags, prg_logits, prg_tags, opr_logits, scale_logits, \
            top2_nums_pred, top2_order_logits, top2_order_logits_bw, top2_order_tags,\
            num_i, gold_i, _ = self.model(
                self._prepare_model_input(batch[12:]), 
                tbl_masks, tbl_nums, tbl_cell_idx,
                prg_masks, prg_nums, prg_word_idx, 
                input_tags=input_tags, opr_tags=opr_tags, order_tags=order_tags)
        
        if split in ('train', 'val', 'test'):
            loss, tbl_loss, prg_loss, opr_loss, order_loss, scale_loss, top2_order_tags \
                = self.model.fit(tbl_logits, tbl_tags, prg_logits, prg_tags,
                    opr_logits, opr_tags, scale_logits, scale_tags, 
                    top2_order_logits_bw, top2_order_tags, gold_i)

        sample_idx, tbl_pred, tbl_tags, tbl_nums, tbl_cell_idx, \
            prg_pred, prg_tags, prg_nums, prg_word_idx, opr_pred, opr_tags, \
            scale_pred, scale_tags, top2_nums_pred, top2_order_logits = self._to_cpu(
                sample_idx, type_lens, tbl_logits, tbl_tags, tbl_nums, tbl_cell_idx,
                prg_logits, prg_tags, prg_nums, prg_word_idx, 
                opr_logits, opr_tags, scale_logits, scale_tags,
                top2_nums_pred=top2_nums_pred, top2_order_logits=top2_order_logits)

        answer_pred, answer_tags, preds = self.predict(
            sample_idx, tbl_pred, tbl_nums, tbl_cell_idx,
            prg_pred, prg_nums, prg_word_idx, opr_pred, scale_pred,
            top2_nums_pred, top2_order_logits, num_i, split)

        records = self.record(sample_idx, tbl_logits, prg_logits,
            opr_logits, scale_logits, top2_nums_pred, top2_order_logits, split)
        
        if split in ('train', 'val', 'test'):
            results = self.evaluate(sample_idx, tbl_pred, tbl_tags, tbl_cell_idx,
                prg_pred, prg_tags, prg_word_idx, answer_pred, answer_tags,
                opr_pred, opr_tags, scale_pred, scale_tags, split)

            return {
                'loss': loss,
                'loss_tbl': cpu(tbl_loss),
                'loss_prg': cpu(prg_loss),
                'loss_opr': cpu(opr_loss),
                'loss_order': cpu(order_loss),
                'loss_scale': cpu(scale_loss),
                'preds': preds,
                'records': records,
                'results': results}
        
        else:
            return {'preds': preds, 'records': records,}

    def _to_cpu(
        self, 
        sample_idx: torch.LongTensor,
        type_lens: torch.LongTensor,
        tbl_logits: torch.FloatTensor,
        tbl_tags: torch.LongTensor,
        tbl_nums: torch.FloatTensor,
        tbl_cell_idx: torch.LongTensor,
        prg_logits: torch.FloatTensor,
        prg_tags: torch.LongTensor,
        prg_nums: torch.FloatTensor,
        prg_word_idx: torch.LongTensor,
        opr_logits: torch.FloatTensor,
        opr_tags: torch.LongTensor,
        scale_logits: torch.FloatTensor,
        scale_tags: torch.LongTensor,
        top2_nums_pred: torch.FloatTensor = None,
        top2_order_logits: torch.FloatTensor = None) -> tuple:
        
        cpu = lambda x: x.detach().cpu().numpy()

        tbl_pred = cpu(self._predict_seq(tbl_logits, tbl_cell_idx))
        prg_pred = cpu(self._predict_seq(prg_logits, prg_word_idx, type_lens))

        tbl_tags = cpu(reduce_max(tbl_tags, tbl_cell_idx))
        prg_tags = cpu(reduce_max(prg_tags, prg_word_idx))

        opr_pred = cpu(torch.argmax(opr_logits, dim=-1))
        scale_pred = cpu(torch.argmax(scale_logits, dim=-1))

        opr_tags = cpu(opr_tags)
        scale_tags = cpu(scale_tags)

        sample_idx = cpu(sample_idx)
        tbl_nums = cpu(tbl_nums)
        prg_nums = cpu(prg_nums)
        tbl_cell_idx = cpu(tbl_cell_idx)
        prg_word_idx = cpu(prg_word_idx)

        top2_nums_pred = cpu(top2_nums_pred) if top2_nums_pred is not None else None
        top2_order_logits = cpu(top2_order_logits) if top2_order_logits is not None else None

        return sample_idx, tbl_pred, tbl_tags, tbl_nums, tbl_cell_idx, \
            prg_pred, prg_tags, prg_nums, prg_word_idx, opr_pred, opr_tags, \
            scale_pred, scale_tags, top2_nums_pred, top2_order_logits
