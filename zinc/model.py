# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
)

from modules import GraphormerGraphEncoder

logger = logging.getLogger(__name__)


class GraphormerModelSlim(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = GraphormerEncoder(args)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=False
                )
            else:
                raise NotImplementedError

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, targets=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb  # with_flag取决于perturb是否传入
        )  # inner_states: 13 * [n_node+1, n_graph, n_hidden]; graph_rep: [n_graph, n_hidden]
        x = inner_states[-1].transpose(0, 1)  # [n_graph, n_node+1, n_hidden]

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)  # [n_graph, n_node+1, 1] 全连接
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        loss = None
        if targets is not None:
            logits = x[:, 0, :]  # [n_graph, 1]
            loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        return x, loss
