# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        contrastive_loss=False,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before), 
            num_encoder_layers, 
            nn.LayerNorm(d_model) if normalize_before else None,
            pass_pos_and_query=pass_pos_and_query)

        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before), 
            num_decoder_layers, 
            nn.LayerNorm(d_model), 
            return_intermediate=return_intermediate_dec,
            pass_pos_and_query=pass_pos_and_query
        )

        self.cls_embed = nn.Embedding(1, d_model) if contrastive_loss else None

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, src, text_memory_resized, mask, text_attention_mask, query_embed, pos_embed: torch.Tensor):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        device = src.device
        if self.cls_embed is not None:
            # We add a CLS token to the image features, to be used for contrastive loss
            src = torch.cat((self.cls_embed.weight.view(1, 1, -1).repeat(1, bs, 1), src))
            mask = torch.cat((torch.zeros(bs, 1).bool().to(device), mask), dim=1)
            pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

        # Concat image & text on the sequence dimension
        src = torch.cat([src, text_memory_resized], dim=0)
        mask = torch.cat([mask, text_attention_mask], dim=1)
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)

        # cross-attend image / text
        img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        text_memory = img_memory[-len(text_memory_resized):]

        assert img_memory.shape[1] == text_memory.shape[1] == query_embed.shape[1]  # batch
        return {
            "text_memory_resized": text_memory_resized,
            "text_memory": text_memory,
            "img_memory": img_memory,
            "mask": mask,
            "text_attention_mask": text_attention_mask,
            "pos_embed": pos_embed,
            "query_embed": query_embed,
        }

    def _decode(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        text_memory,
        img_memory,
        text_attention_mask,
    ):
        assert img_memory.shape[1] == text_memory.shape[1] == query_embed.shape[1]  # batch
        hs = self.decoder(
            query_embed,
            img_memory,
            text_memory,
            memory_key_padding_mask=mask,
            text_memory_key_padding_mask=text_attention_mask,
            pos=pos_embed,
        )
        return hs.transpose(1, 2)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
        text_features=None,
        encode_and_save=True,
        **memory_cache,  # catch extra params from memory_cache
    ):
        if encode_and_save:
            return self._encode(
                src=src,
                mask=mask,
                query_embed=query_embed,
                pos_embed=pos_embed,
                text_features=text_features,
            )

        else:
            return self._decode(
                src=src,
                mask=mask,
                query_embed=query_embed,
                pos_embed=pos_embed,
                text_memory=text_memory,
                img_memory=img_memory,
                text_attention_mask=text_attention_mask,
            )


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, pass_pos_and_query=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pass_pos_and_query = pass_pos_and_query

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # add positional embedding at the beginning?
        if not self.pass_pos_and_query:
            src = src + 0.1 * pos
            pos = None

        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, 
                 return_intermediate=False, pass_pos_and_query=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.pass_pos_and_query = pass_pos_and_query

    def forward(
        self,
        query_pos,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # add positional embedding at the beginning? or to every layer?
        if self.pass_pos_and_query:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = query_pos
            pos = query_pos = None

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to text
        # tgt2 = self.cross_attn_text(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=text_memory,
        #     value=text_memory,
        #     attn_mask=None,
        #     key_padding_mask=text_memory_key_padding_mask,
        # )[0]
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        assert False, "not implemented yet"
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        contrastive_loss=args.contrastive_loss,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
