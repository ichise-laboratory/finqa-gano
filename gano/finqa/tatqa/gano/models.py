import torch
from allennlp.nn import util
from torch_geometric.data import Batch

from gano.finqa.tatqa.geer.models import GeerModel, GeerLightning
from gano.finqa.tatqa.noc.models import NocModel, NocLightning


class GanoModel(GeerModel, NocModel):
    def forward(
        self,
        model_input: dict,
        tbl_masks: torch.LongTensor,
        prg_masks: torch.LongTensor,
        graphs: Batch,
        input_tags: torch.LongTensor = None,
        order_masks: torch.LongTensor = None) -> tuple:

        mask = lambda e, m, v: util.replace_masked_values(e, m.unsqueeze(-1).bool(), v)
        log_softmax = lambda l, m: util.masked_log_softmax(l, mask=m.unsqueeze(-1).bool())

        outputs = self.encoder(**model_input)
        seq_emb = outputs[0]
        seq_len = seq_emb.shape[1]
        batch_size = seq_emb.shape[0]

        prg_emb = mask(seq_emb, prg_masks, 0)

        if self.use_lstm:
            prg_emb, _ = self.lstm(prg_emb)
            
        prg_logits = self.prg_classifier(prg_emb)
        prg_logits = log_softmax(prg_logits, prg_masks)

        cls_emb = seq_emb[:, 0, :]
        tbl_emb = mask(seq_emb, tbl_masks, 0)

        seq_emb = seq_emb.view(batch_size * seq_len, -1)
        seq_emb = self.gnn(seq_emb, graphs.edge_index)

        if self.gnn_droupout is not None:
            if self.gnn_cls == 'pan':
                seq_emb = self.gnn_droupout(seq_emb[0])
            else:
                seq_emb = self.gnn_droupout(seq_emb)

        seq_emb = seq_emb.view(batch_size, seq_len, -1)

        tbl_gnn_emb = mask(seq_emb, tbl_masks, 0)
        tbl_logits = self.tbl_classifier(tbl_gnn_emb)
        tbl_logits = log_softmax(tbl_logits, tbl_masks)

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


class GanoLightning(GeerLightning, NocLightning):
    def init_model_cls(self) -> None:
        super().init_model_cls()
        self.model_cls = GanoModel
    
    def predict_step(self, batch: tuple, batch_idx: int):
        return self._step(batch, batch_idx, 'predict')
    
    def train_test_step(self, batch: tuple, batch_idx: int, split: str):
        return self._step(batch, batch_idx, split)
    
    def _step(self, batch: tuple, batch_idx: int, split: str) -> dict:
        cpu = lambda x: x.detach().cpu().numpy()

        sample_idx, tbl_masks, tbl_nums, tbl_cell_idx, tbl_nums_map, \
            prg_masks, prg_nums, prg_word_idx, prg_nums_map, \
            input_tags, opr_tags, order_tags, order_nums, order_masks, \
            scale_tags, type_lens, deriv_nums, deriv_pos, graphs = batch[:19]
        
        tbl_logits, tbl_tags, prg_logits, prg_tags, opr_logits, \
            scale_logits, order_logits, order_train_logits, _ = self.model(
                self._prepare_model_input(batch[19:]),
                tbl_masks, prg_masks, graphs, 
                input_tags=input_tags, order_masks=order_masks)

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
