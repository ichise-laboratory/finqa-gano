import logging

import torch
import torch.nn as nn
from allennlp.nn import util
from torch_geometric.data import Batch
from torch_geometric.nn import EGConv, GATConv, PANConv, SAGEConv, SuperGATConv

from gano.finqa.tatqa.tagop.models import TagOpLightning, TagOpModel
from gano.finqa.tatqa.tagop.nn import FFNLayer, reduce_max, reduce_mean


class GeerModel(TagOpModel):
    def __init__(self, *args, gnn_cls: str = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.gnn_cls = gnn_cls
        self.tag_classifier = None
        self.tbl_classifier = FFNLayer(
            self.hidden_size, self.hidden_size, self.num_labels, self.dropout_prob)
        self.prg_classifier = FFNLayer(
            self.hidden_size, self.hidden_size, self.num_labels, self.dropout_prob)
        
        hidden_size = self.config.hidden_size

        if gnn_cls == 'eg':
            logging.info('Using EGConv for GNN.')
            self.gnn = EGConv(hidden_size, hidden_size)
            self.gnn_droupout = nn.Dropout(self.dropout_prob)
        
        elif gnn_cls == 'gat':
            logging.info('Using GATConv for GNN.')
            self.gnn = GATConv(hidden_size, hidden_size, 
                dropout=self.dropout_prob, heads=8, concat=False)
            self.gnn_droupout = None

        elif gnn_cls == 'pan':
            logging.info('Using PANConv for GNN.')
            self.gnn = PANConv(hidden_size, hidden_size, 4)
            self.gnn_droupout = nn.Dropout(self.dropout_prob)

        elif gnn_cls == 'sage':
            logging.info('Using SAGEConv for GNN.')
            self.gnn = SAGEConv(hidden_size, hidden_size)
            self.gnn_droupout = nn.Dropout(self.dropout_prob)
        
        elif gnn_cls == 'sgat':
            logging.info('Using SuperGATConv for GNN.')
            self.gnn = SuperGATConv(hidden_size, hidden_size, dropout=self.dropout_prob)
            self.gnn_droupout = nn.Dropout(self.dropout_prob)
    
    def forward(
        self,
        model_input: dict,
        tbl_masks: torch.LongTensor,
        tbl_nums: torch.FloatTensor,
        tbl_cell_idx: torch.LongTensor,
        prg_masks: torch.LongTensor,
        prg_nums: torch.FloatTensor,
        prg_word_idx: torch.LongTensor,
        graphs: Batch,
        input_tags: torch.LongTensor = None,
        opr_tags: torch.LongTensor = None,
        order_tags: torch.LongTensor = None) -> tuple:

        mask = lambda e, m, v: util.replace_masked_values(e, m.unsqueeze(-1).bool(), v)
        log_softmax = lambda l, m: util.masked_log_softmax(l, mask=m.unsqueeze(-1).bool())

        outputs = self.encoder(**model_input)
        seq_emb = outputs[0]
        seq_len = seq_emb.shape[1]
        batch_size = seq_emb.shape[0]
        device = tbl_masks.device

        prg_emb = mask(seq_emb, prg_masks, 0)
        
        if self.use_lstm:
            prg_emb, _ = self.lstm(prg_emb)

        prg_logits = self.prg_classifier(prg_emb)
        prg_logits = log_softmax(prg_logits, prg_masks)

        cls_emb = seq_emb[:, 0, :]
        seq_emb = seq_emb.view(batch_size * seq_len, -1)
        seq_emb = self.gnn(seq_emb, graphs.edge_index)

        if self.gnn_droupout is not None:
            if self.gnn_cls == 'pan':
                seq_emb = self.gnn_droupout(seq_emb[0])
            else:
                seq_emb = self.gnn_droupout(seq_emb)

        seq_emb = seq_emb.view(batch_size, seq_len, -1)

        tbl_emb = mask(seq_emb, tbl_masks, 0)
        tbl_logits = self.tbl_classifier(tbl_emb)
        tbl_logits = log_softmax(tbl_logits, tbl_masks)

        gnn_cls_emb = seq_emb[:, 0, :]

        if input_tags is not None:
            tbl_tags = util.replace_masked_values(input_tags.float(), tbl_masks.bool(), 0)
            prg_tags = util.replace_masked_values(input_tags.float(), prg_masks.bool(), 0)
        
        else:
            tbl_tags = prg_tags = None

        prg_reduce_mean = torch.mean(prg_emb, dim=1)
        tbl_reduce_mean = torch.mean(tbl_emb, dim=1)
        cls_tbl_prg_emb = torch.cat((gnn_cls_emb, tbl_reduce_mean, prg_reduce_mean), dim=-1)

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


class GeerLightning(TagOpLightning):
    def __init__(self, gnn_cls: str = 'sage', *args, **kwargs):
        self.gnn_cls = gnn_cls
        super().__init__(*args, **kwargs)

    def init_model(self, model_params: dict = None) -> None:
        if model_params is None: 
            model_params = {'gnn_cls': self.gnn_cls}
        else:
            model_params['gnn_cls'] = self.gnn_cls
        
        super().init_model(model_params)

    def init_model_cls(self) -> None:
        super().init_model_cls()
        self.model_cls = GeerModel
    
    def predict_step(self, batch: tuple, batch_idx: int):
        return self._step(batch, batch_idx, 'predict')

    def train_test_step(self, batch: tuple, batch_idx: int, split: str):
        return self._step(batch, batch_idx, split)

    def _step(self, batch: tuple, batch_idx: int, split: str) -> dict:
        cpu = lambda x: x.detach().cpu().numpy()

        sample_idx, tbl_masks, tbl_nums, tbl_cell_idx, \
            prg_masks, prg_nums, prg_word_idx, \
            input_tags, opr_tags, order_tags, scale_tags, type_lens, graphs = batch[:13]
        
        tbl_logits, tbl_tags, prg_logits, prg_tags, opr_logits, scale_logits, \
            top2_nums_pred, top2_order_logits, top2_order_logits_bw, top2_order_tags,\
            num_i, gold_i, _ = self.model(
                self._prepare_model_input(batch[13:]), 
                tbl_masks, tbl_nums, tbl_cell_idx,
                prg_masks, prg_nums, prg_word_idx, 
                graphs, input_tags=input_tags, 
                opr_tags=opr_tags, order_tags=order_tags)
        
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


class GeerLightning(GeerLightning):
    pass
