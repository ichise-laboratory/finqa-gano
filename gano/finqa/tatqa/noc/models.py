import numpy as np
import torch
import torch.nn as nn
from allennlp.nn import util

from gano.finqa.tatqa.tagop.configs import OPERATORS, SCALES
from gano.finqa.tatqa.tagop.models import TagOpModel, TagOpLightning
from gano.finqa.tatqa.tagop.nn import FFNLayer
from gano.finqa.tatqa.tagop.utils import get_tbl_spans, get_prg_spans, get_nums


class NocModel(TagOpModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.order_classifier = FFNLayer(
            self.hidden_size, self.hidden_size, 3, self.dropout_prob)
        self.order_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        model_input: dict,
        tbl_masks: torch.LongTensor,
        prg_masks: torch.LongTensor,
        input_tags: torch.LongTensor = None,
        order_masks: torch.LongTensor = None) -> tuple:

        mask = lambda e, m, v: util.replace_masked_values(e, m.unsqueeze(-1).bool(), v)
        log_softmax = lambda l, m: util.masked_log_softmax(l, mask=m.unsqueeze(-1).bool())

        outputs = self.encoder(**model_input)
        seq_emb = outputs[0]
        cls_emb = seq_emb[:, 0, :]

        tbl_emb = mask(seq_emb, tbl_masks, 0)
        tbl_logits = self.tag_classifier(tbl_emb)
        tbl_logits = log_softmax(tbl_logits, tbl_masks)

        prg_emb = mask(seq_emb, prg_masks, 0)

        if self.use_lstm:
            prg_emb, _ = self.lstm(prg_emb)
            
        prg_logits = self.tag_classifier(prg_emb)
        prg_logits = log_softmax(prg_logits, prg_masks)

        if input_tags is not None:
            tbl_tags = util.replace_masked_values(input_tags.float(), tbl_masks.bool(), 0)
            prg_tags = util.replace_masked_values(input_tags.float(), prg_masks.bool(), 0)
        
        else:
            tbl_tags = prg_tags = None
        
        prg_reduce_mean = torch.mean(prg_emb, dim=1)
        tbl_reduce_mean = torch.mean(tbl_emb, dim=1)
        cls_tbl_prg_emb = torch.cat((cls_emb, tbl_reduce_mean, prg_reduce_mean), dim=-1)

        opr_logits = self.opr_classifier(cls_emb)
        scale_logits = self.scale_classifier(cls_tbl_prg_emb)
        order_logits = self.order_classifier(seq_emb)
        order_logits = util.masked_log_softmax(order_logits, None)

        if order_masks is not None:
            order_emb = mask(seq_emb, order_masks, 0)
            order_train_logits = self.order_classifier(order_emb)
            order_train_logits = log_softmax(order_train_logits, order_masks)
        
        else:
            order_train_logits = None

        return tbl_logits, tbl_tags, prg_logits, prg_tags, opr_logits, \
            scale_logits, order_logits, order_train_logits, outputs
    
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
        order_train_logits: torch.FloatTensor,
        order_tags: torch.LongTensor) -> tuple:

        tbl_logits = tbl_logits.transpose(1, 2)
        prg_logits = prg_logits.transpose(1, 2)

        tbl_loss = self.NLLLoss(tbl_logits, tbl_tags.long())
        prg_loss = self.NLLLoss(prg_logits, prg_tags.long())
        opr_loss = self.opr_loss_fn(opr_logits, opr_tags)
        scale_loss = self.scale_loss_fn(scale_logits, scale_tags)

        loss = tbl_loss + prg_loss + opr_loss + scale_loss
        order_loss = None

        if order_train_logits is not None:
            order_rel_logits, order_rel_tags = [], []

            for i in range(order_train_logits.shape[0]):
                if opr_tags[i] in self.opr_order_cls:
                    order_rel_logits.append(order_train_logits[i])
                    order_rel_tags.append(order_tags[i])
            
            if len(order_rel_tags):
                order_rel_logits = torch.stack(order_rel_logits)
                order_rel_tags = torch.stack(order_rel_tags)
                order_rel_logits = order_rel_logits.transpose(1, 2)

                order_loss = self.order_loss_fn(order_rel_logits, order_rel_tags)
                loss += order_loss
        
        return loss, tbl_loss, prg_loss, opr_loss, scale_loss, order_loss


class NocLightning(TagOpLightning):
    METRIC_KEYS = ('opr', 'scale', 'order', 'em.ext.hyb', 'f1.ext.hyb', 
        'em.ext.tbl', 'f1.ext.tbl', 'em.ext.prg', 'f1.ext.prg')
        
    def init_model_cls(self) -> None:
        super().init_model_cls()
        self.model_cls = NocModel

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
        order_pred: list,
        split: str) -> tuple:
        
        preds = {}
        answer_pred, answer_tags = [], []
        batch_size = tbl_pred.shape[0]

        texts = self.dataset.texts[split]
        offsets = self.dataset.offsets[split]

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
            
            elif opr_pred[i] in self.opr_order_cls and order_pred[i] is not None:
                num_a, num_b = order_pred[i]
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
        split: str) -> dict:

        cpu = lambda x: x.detach().cpu().numpy()
        records = {}

        for i in range(sample_idx.shape[0]):
            qid = self.dataset.samples[split][sample_idx[i]]['question']['uid']
            records[qid] = {
                'tbl_logits': cpu(tbl_logits),
                'prg_logits': cpu(prg_logits),
                'opr_logits': cpu(opr_logits),
                'scale_logits': cpu(scale_logits)}
        
        return records

    def predict_step(self, batch: tuple, batch_idx: int):
        return self._step(batch, batch_idx, 'predict')
    
    def train_test_step(self, batch: tuple, batch_idx: int, split: str):
        return self._step(batch, batch_idx, split)
    
    def _evaluate_order(
        self,
        sample_idx: np.ndarray,
        opr_tags: np.ndarray,
        order_train_pred: list,
        order_nums: np.ndarray,
        results: dict,
        split: str) -> dict:
        
        for i in range(len(sample_idx)):
            if opr_tags[i] in self.opr_order_cls:
                qid = self.dataset.ids_map[split][sample_idx[i]]

                if order_train_pred[i][0] == order_nums[i, 0] and \
                    order_train_pred[i][1] == order_nums[i, 1]:
                    results[qid]['order'] = 1
                
                else:
                    results[qid]['order'] = 0

        return results
    
    def _get_pos_num(
        self,
        tbl_nums: torch.FloatTensor,
        tbl_cell_idx: torch.LongTensor,
        prg_nums: torch.FloatTensor,
        prg_word_idx: torch.LongTensor,
        pos: torch.LongTensor) -> torch.LongTensor:

        if tbl_cell_idx[pos] > 0:
            return tbl_nums[tbl_cell_idx[pos]]
        
        elif prg_word_idx[pos] > 0:
            return prg_nums[prg_word_idx[pos]]
        
        else:
            return None

    def _predict_order(
        self,
        tbl_pred: np.ndarray,
        tbl_nums: np.ndarray,
        tbl_nums_map: torch.LongTensor,
        prg_pred: np.ndarray,
        prg_nums: np.ndarray,
        prg_nums_map: torch.LongTensor,
        opr_pred: np.ndarray,
        order_logits: torch.FloatTensor) -> list:
        
        cpu = lambda x: x.detach().cpu().numpy()
        order_pred = [None] * order_logits.shape[0]

        for i in range(order_logits.shape[0]):
            if opr_pred[i] in self.opr_order_cls:
                st_pred, nd_pred = [], []

                tbl_pred_nums, tbl_pos = get_nums(tbl_pred[i], tbl_nums[i])
                prg_pred_nums, prg_pos = get_nums(prg_pred[i], prg_nums[i])
                pred_nums = tbl_pred_nums + prg_pred_nums

                if len(pred_nums) >= 2:
                    tbl_pos = [cpu(tbl_nums_map[i, p - 1]) for p in tbl_pos]
                    prg_pos = [cpu(prg_nums_map[i, p - 1]) for p in prg_pos]

                    for j, pos in enumerate(tbl_pos + prg_pos):
                        num_logits = order_logits[i, pos[0]:pos[1]].transpose(0, 1)
                        logits = cpu(num_logits.max(dim=-1).values)
                        st_pred.append((logits[1], pred_nums[j]))
                        nd_pred.append((logits[2], pred_nums[j]))

                    st_pred = sorted(st_pred, key=lambda x: x[0], reverse=True)
                    nd_pred = sorted(nd_pred, key=lambda x: x[0], reverse=True)

                    order_pred[i] = (st_pred[0][1], nd_pred[0][1])
    
        return order_pred


    def _predict_train_order(
        self,
        opr_tags: torch.LongTensor,
        order_train_logits: torch.FloatTensor,
        deriv_nums: torch.FloatTensor,
        deriv_pos: torch.LongTensor,) -> list:

        cpu = lambda x: x.detach().cpu().numpy()
        order_pred = []

        for i in range(order_train_logits.shape[0]):
            if opr_tags[i] in self.opr_order_cls:
                pos = cpu(deriv_pos[i])
                logits = []

                for j in (0, 1):
                    num_logits = order_train_logits[i, pos[j, 0]:pos[j, 1]].transpose(0, 1)
                    logits.append(cpu(num_logits.max(dim=-1).values))

                if logits[0][1] > logits[1][1]:
                    order_pred.append((deriv_nums[i, 0], deriv_nums[i, 1]))
                    
                else:
                    order_pred.append((deriv_nums[i, 1], deriv_nums[i, 0]))

            else:
                order_pred.append(None)
        
        return order_pred

    def _step(self, batch: tuple, batch_idx: int, split: str) -> dict:
        cpu = lambda x: x.detach().cpu().numpy()

        sample_idx, tbl_masks, tbl_nums, tbl_cell_idx, tbl_nums_map, \
            prg_masks, prg_nums, prg_word_idx, prg_nums_map, \
            input_tags, opr_tags, order_tags, order_nums, order_masks, \
            scale_tags, type_lens, deriv_nums, deriv_pos = batch[:18]
        
        tbl_logits, tbl_tags, prg_logits, prg_tags, opr_logits, \
            scale_logits, order_logits, order_train_logits, _ = self.model(
                self._prepare_model_input(batch[18:]),
                tbl_masks, prg_masks, input_tags=input_tags, 
                order_masks=order_masks)

        if split in ('train', 'val', 'test'):
            loss, tbl_loss, prg_loss, opr_loss, scale_loss, order_loss \
                = self.model.fit(tbl_logits, tbl_tags, prg_logits, prg_tags,
                    opr_logits, opr_tags, scale_logits, scale_tags,
                    order_train_logits, order_tags)

            order_train_pred = self._predict_train_order(
                opr_tags, order_train_logits, deriv_nums, deriv_pos)
        
        sample_idx, tbl_pred, tbl_tags, tbl_nums, tbl_cell_idx, \
            prg_pred, prg_tags, prg_nums, prg_word_idx, opr_pred, opr_tags, \
            scale_pred, scale_tags, _, _ = self._to_cpu(
                sample_idx, type_lens, 
                tbl_logits, tbl_tags, tbl_nums, tbl_cell_idx,
                prg_logits, prg_tags, prg_nums, prg_word_idx, 
                opr_logits, opr_tags, 
                scale_logits, scale_tags)

        order_pred = self._predict_order(
            tbl_pred, tbl_nums, tbl_nums_map,
            prg_pred, prg_nums, prg_nums_map,
            opr_pred, order_logits)
        
        answer_pred, answer_tags, preds = self.predict(
            sample_idx, tbl_pred, tbl_nums, tbl_cell_idx,
            prg_pred, prg_nums, prg_word_idx, opr_pred, 
            scale_pred, order_pred, split)

        records = self.record(sample_idx, tbl_logits, prg_logits,
            opr_logits, scale_logits, split)
        
        if split in ('train', 'val', 'test'):
            results = self.evaluate(sample_idx, tbl_pred, tbl_tags, tbl_cell_idx,
                prg_pred, prg_tags, prg_word_idx, answer_pred, answer_tags,
                opr_pred, opr_tags, scale_pred, scale_tags, split)

            results = self._evaluate_order(
                sample_idx, opr_tags, order_train_pred, order_nums, results, split)

            return {
                'loss': loss,
                'loss_tbl': cpu(tbl_loss),
                'loss_prg': cpu(prg_loss),
                'loss_opr': cpu(opr_loss),
                'loss_order': cpu(order_loss) if order_loss is not None else None,
                'loss_scale': cpu(scale_loss),
                'preds': preds,
                'records': records,
                'results': results}
        
        else:
            return {'preds': preds, 'records': records}
