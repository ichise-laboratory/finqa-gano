import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class FFNLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        intermediate_dim: int, 
        output_dim: int, 
        dropout: float, 
        layer_norm: bool = True):

        super().__init__()
        self.ln = nn.LayerNorm(intermediate_dim) if layer_norm else None
        self.dropout_func = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
    
    def forward(self, input: torch.FloatTensor):
        inter = self.fc1(self.dropout_func(input))
        inter_act = F.gelu(inter)

        if self.ln:
            inter_act = self.ln(inter_act)
        
        return self.fc2(inter_act)


def _flatten_idx(idx, seq_len):
    batch_size = idx.shape[0]
    offset = torch.arange(start=0, end=batch_size, device=idx.device) * seq_len
    offset = offset.view(batch_size, 1)
    return (idx + offset).view(-1), batch_size * seq_len


def reduce_idx(tensor, idx, reduce, dim=0):
        batch_size = tensor.shape[0]
        seq_len = tensor.shape[1]
        unit_size = tensor.shape[2] if len(tensor.shape) == 3 else None
        flat_idx, idx_size = _flatten_idx(idx, seq_len)

        if unit_size is not None:
            flat_tensor = tensor.reshape(batch_size * seq_len, unit_size)
        else:
            flat_tensor = tensor.reshape(batch_size * seq_len)

        idx_max = scatter(
            flat_tensor, 
            flat_idx.type(torch.long), 
            dim=dim, 
            dim_size=idx_size,
            reduce=reduce)

        if unit_size is not None:
            return idx_max.view(batch_size, -1, unit_size)
        else:
            return idx_max.view(batch_size, -1)
    

def reduce_max(emb, idx, dim=0):
    return reduce_idx(emb, idx, 'max', dim)


def reduce_mean(emb, idx, dim=0):
    return reduce_idx(emb, idx, 'mean', dim)
