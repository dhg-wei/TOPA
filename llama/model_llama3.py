# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Embedding, Linear
import torch
import pickle
import timm.models.hub as timm_hub
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


@dataclass
class ModelArgs_llama3:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

    adapter_len: int=10
    adapter_layer: int=30


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
    

class Attention(nn.Module):
    def __init__(self, args: ModelArgs_llama3):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats
        self.n_kv_heads = args.n_kv_heads
        model_parallel_size=1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
    
        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.gate1 = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.gate2 = torch.nn.Parameter(torch.ones(1, self.n_local_heads, 1, 1) * -args.bias)
        self.llamatype = args.llamatype
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:            
            adapter_scores = F.softmax(scores[..., :adapter_len].float(), dim=-1).type_as(xq) * self.gate1.tanh().type(self.llamatype)
            if video_start is not None:
                vt_scores = scores[..., adapter_len:].clone()
                vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] = \
                    vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] + self.gate2.type(self.llamatype)
                vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            else:
                vt_scores = F.softmax(scores[..., adapter_len:], dim=-1)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs_llama3):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)



    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, video_start)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer_llama3(nn.Module):
    def __init__(self, params: ModelArgs_llama3, args):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias
        self.args = args

        if self.args.llama2:
            self.llamatype = torch.bfloat16
            params.llamatype = torch.bfloat16
        else:
            self.llamatype = torch.half
            params.llamatype = torch.half
            
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats


        self.tok_embeddings = Embedding(params.vocab_size, params.dim)
        self.adapter_query = Embedding(params.adapter_len * params.adapter_layer, params.dim)

        clip_feature_dim=768
        self.visual_proj = Linear(clip_feature_dim, params.dim, bias=False)
        self.temporal_emb = Embedding(self.max_feats, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.vqa_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.vaq_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.qav_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.inference_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        self.video_label = torch.arange(1, self.max_feats)
        self.tau = args.tau
        if self.args.memory:
            with open('./data/textvid/memory.pkl','rb') as f:
                self.memory = pickle.load(f).float()[:1000000].cuda()
        
        self.visual_proj = Linear(clip_feature_dim, params.dim, bias=False)

                


    def re_init_freqs(self,max_seq_len):
        self.params.max_seq_len=max_seq_len
        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
        
    def forward(self, data, inference=False,mode='vqa'):

        video = data['video'].cuda()
        # video = video/video.norm(dim=-1,keepdim=True)
        if self.args.memory and inference==True:
            sim = video@self.memory.T

            sim = (sim*100).softmax(dim=-1)
            video = sim@self.memory
            video = video/video.norm(dim=-1,keepdim=True)
        if self.args.onlyqa:
            video=video*0

        vqa_id, vaq_id, qav_id = data['text_id']['vqa'].cuda(), data['text_id']['vaq'].cuda(), data['text_id']['qav'].cuda()
        vqa_label, vaq_label, qav_label = data['label']['vqa'].cuda(), data['label']['vaq'].cuda(), data['label']['qav'].cuda()
        vqa_video_start, vaq_video_start, qav_video_index = data['video_start']['vqa'][0], data['video_start']['vaq'][0], data['video_index']['qav'].cuda()
        
        bsz, n_options, seqlen = vqa_id.shape
        vqa_id, vaq_id = vqa_id.reshape(-1, seqlen), vaq_id.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label.reshape(-1, seqlen), vaq_label.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label[:, 1:].flatten(), vaq_label[:, 1:].flatten()
        
        qav_id = qav_id.reshape(-1, seqlen)
        qav_label = qav_label.reshape(-1, seqlen)
        qav_video_mask = qav_label.ge(0)
        qav_label = qav_label[:, 1:].flatten()
        
        
        with torch.no_grad():
            vqa_h = self.tok_embeddings(vqa_id)
            
            if self.args.vaq and not inference:
                vaq_h = self.tok_embeddings(vaq_id)
            if self.args.openvqa_eval:
                vaq_h = self.tok_embeddings(vaq_id)
            if self.args.qav and not inference:
                qav_h = self.tok_embeddings(qav_id)
            
        freqs_cis = self.freqs_cis.to(vqa_h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=vqa_h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(vqa_h)
        start_pos = 0
        vqa_loss, vaq_loss, qav_loss = torch.tensor([0]).cuda(),torch.tensor([0]).cuda(), torch.tensor([0]).cuda()
        
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        
        _video_feature = self.visual_proj(video)

        if inference:
            _video_feature = _video_feature.unsqueeze(1).repeat(1, n_options, 1, 1).view(-1, _video_feature.shape[-2], _video_feature.shape[-1])
        video_feature = (_video_feature + self.temporal_emb.weight[None, :, :]).type(self.llamatype)
        
        
        if mode == 'vqa':
             
            vqa_h = vqa_h.clone()
            vqa_h[:, vqa_video_start:vqa_video_start+self.max_feats] = video_feature

            for i, layer in enumerate(self.layers[-1 * self.adapter_layer:]):
                vqa_h = layer(vqa_h, start_pos, freqs_cis, mask, adapter[i].type(self.llamatype), vqa_video_start)
            
            vqa_h = self.norm(vqa_h)
            vqa_output = self.output(vqa_h)
            vqa_output = vqa_output[:, :-1, :].reshape(-1, self.vocab_size)
            vqa_loss = self.vqa_criterion(vqa_output, vqa_label)

        if mode =='caption':
            vaq_h = vaq_h.clone()
            vaq_h[:, vaq_video_start:vaq_video_start+self.max_feats] = video_feature
                
            for i, layer in enumerate(self.layers[-1 * self.adapter_layer:]):
                vaq_h = layer(vaq_h, start_pos, freqs_cis, mask, adapter[i].type(self.llamatype), vaq_video_start)
            vaq_h = self.norm(vaq_h)
            vaq_output = self.output(vaq_h)
            vaq_output = vaq_output[:, :-1, :].reshape(-1, self.vocab_size)
            vaq_loss = self.vaq_criterion(vaq_output, vaq_label)
            
        if inference:
            if self.args.openvqa_eval:
                logits = self.inference_criterion(vaq_output, vaq_label)
                logits = logits.reshape(bsz, n_options, -1)
            else:
                logits = self.inference_criterion(vqa_output, vqa_label)
                logits = logits.reshape(bsz, n_options, -1)      
            return logits
        else:
            return vqa_loss, vaq_loss, qav_loss
