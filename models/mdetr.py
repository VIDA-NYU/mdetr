# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor

from .backbone import build_backbone
# from .matcher import build_matcher
from .segmentation import DETRsegm
from .transformer import build_transformer
from .criterion import build_criterion
from .position_encoding import build_position_encoding
from .text_encoder import build_text_encoder


class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """
    qa_head = None

    def __init__(
        self,
        backbone,
        text_encoder,
        transformer,
        num_classes,
        num_queries,
        pos_encode=None,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        contrastive_align_loss=False,
        qa_dataset: Optional[str] = None,
        split_qa_heads=True,
        predict_final=False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_loss: If true, perform image-text contrastive learning
            contrastive_align_loss: If true, perform box - token contrastive learning
            qa_dataset: If not None, train a QA head for the target dataset (CLEVR or GQA)
            split_qa_heads: If true, use several head for each question type
            predict_final: If true, will predict if a given box is in the actual referred set.
                           Useful for CLEVR-Ref+ only currently.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.backbone = backbone
        self.text_encoder = text_encoder

        self.pos_encode = pos_encode
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if qa_dataset is not None:
            nb_heads = 6 if qa_dataset == "gqa" else 4
            self.qa_embed = nn.Embedding(nb_heads if split_qa_heads else 1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # contrastive loss layers

        self.aux_loss = aux_loss
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False)
            self.contrastive_projection_text = nn.Linear(
                self.transformer.text_encoder.config.hidden_size, contrastive_hdim, bias=False
            )
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads
        if qa_dataset is not None:
            if qa_dataset == "gqa":
                if split_qa_heads:
                    self.qa_head = SplitGQAAnswerHead(hidden_dim)
                else:
                    self.qa_head = GQAAnswerHead(hidden_dim)
            elif qa_dataset == "clevr":
                assert not split_qa_heads, "Clevr QA is not supported with unified head"
                self.qa_head = ClevrAnswerHead(hidden_dim)
            else:
                assert False, f"Invalid qa dataset {qa_dataset}"

    def _encode(self, samples: NestedTensor, captions: list[str]|list[torch.Tensor]):
        # encode image
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)
        features = self.backbone(samples)
        src, mask = features[-1].decompose()

        # encode text
        (
            text_memory, text_attention_mask, tokenized, pooled_encoded_text
        ) = self.text_encoder(captions, src.device)

        # positional embedding
        pos_embed = None
        if self.pos_encode is not None:
            pos_embed = self.pos_encode(src).to(src.tensors.dtype)

        # query embedding
        query_embed = self.query_embed.weight
        if self.qa_dataset is not None:
            query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)

        # cross-attend text and image
        memory_cache = self.transformer(
            self.input_proj(src),
            text_memory,
            mask=mask,
            text_attention_mask=text_attention_mask,
            pos_embed=pos_embed,
            query_embed=query_embed,
            encode_and_save=True,
        )
        memory_cache['tokenized'] = tokenized

        if self.contrastive_loss:
            memory_cache["img_pooled_op"] = memory_cache['img_memory'][0]
            memory_cache["text_pooled_op"] = pooled_encoded_text
            memory_cache["text_pooled_op"] = self.contrastive_projection_text(memory_cache["text_pooled_op"])
            memory_cache["img_pooled_op"] = self.contrastive_projection_image(memory_cache["img_pooled_op"])

        return memory_cache

    def _decode(self, img_memory, text_memory, text_memory_resized, text_attention_mask, mask, query_embed, pos_embed, tokenized):
        # cross-attend
        hs = self.transformer(
            mask=mask,
            query_embed=query_embed,
            pos_embed=pos_embed,
            encode_and_save=False,
            text_memory=text_memory_resized,
            img_memory=img_memory,
            text_attention_mask=text_attention_mask,
        )
        out = {}

        # question answering
        if self.qa_head is not None:
            qa_output, hs = self.qa_head(hs)
            out.update(qa_output)

        # predicted boxes
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out.update({
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        })
        
        # predict box is in the actual referred set.
        outputs_isfinal = None
        if self.isfinal_embed is not None:
            outputs_isfinal = self.isfinal_embed(hs)
            out["pred_isfinal"] = outputs_isfinal[-1]

        # box-token contrastive loss
        proj_queries, proj_tokens = None, None
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
            proj_tokens = F.normalize(self.contrastive_align_projection_text(text_memory).transpose(0, 1), p=2, dim=-1)
            out.update({
                "proj_queries": proj_queries[-1],
                "proj_tokens": proj_tokens,
                "tokenized": tokenized,
            })

        # apply loss to each decoder layer
        if self.aux_loss:
            if self.contrastive_align_loss:
                assert proj_tokens is not None and proj_queries is not None
                out["aux_outputs"] = [
                    {
                        "pred_logits": a,
                        "pred_boxes": b,
                        "proj_queries": c,
                        "proj_tokens": proj_tokens,
                        "tokenized": tokenized,
                    }
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                ]
            else:
                out["aux_outputs"] = [
                    {
                        "pred_logits": a,
                        "pred_boxes": b,
                    }
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            if outputs_isfinal is not None:
                assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                for i in range(len(outputs_isfinal[:-1])):
                    out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]
        return out

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if encode_and_save:
            assert memory_cache is None
            return self._encode(samples, captions)
        else:
            assert memory_cache is not None
            return self._decode(**memory_cache)

    # # convenience wrappers so its clear how to call things
    # def encode(self, samples: NestedTensor, captions):
    #     return self(samples, captions)

    # def decode(self, memory_cache):
    #     return self(memory_cache=memory_cache, encode_and_save=False)

    # def full(self, samples: NestedTensor, captions):
    #     memory_cache = self.encode(samples, captions)
    #     out = self.decode(memory_cache)
    #     return dict(memory_cache, **out)



class GQAAnswerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.answer_head = nn.Linear(hidden_dim, 1853)

    def __call__(self, hs):
        out = {}
        answer_embeds = hs[0, :, -1]
        hs = hs[:, :, :-1]
        out["pred_answer"] = self.answer_head(answer_embeds)
        return out, hs


class SplitGQAAnswerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.answer_type_head = nn.Linear(hidden_dim, 5)
        self.answer_rel_head = nn.Linear(hidden_dim, 1594)
        self.answer_obj_head = nn.Linear(hidden_dim, 3)
        self.answer_global_head = nn.Linear(hidden_dim, 111)
        self.answer_attr_head = nn.Linear(hidden_dim, 403)
        self.answer_cat_head = nn.Linear(hidden_dim, 678)

    def __call__(self, hs):
        out = {}
        answer_embeds = hs[0, :, -6:]
        hs = hs[:, :, :-6]
        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
        out["pred_answer_obj"] = self.answer_obj_head(answer_embeds[:, 1])
        out["pred_answer_rel"] = self.answer_rel_head(answer_embeds[:, 2])
        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
        out["pred_answer_cat"] = self.answer_cat_head(answer_embeds[:, 4])
        out["pred_answer_global"] = self.answer_global_head(answer_embeds[:, 5])
        return out, hs

class ClevrAnswerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.answer_type_head = nn.Linear(hidden_dim, 5)
        self.answer_type_head = nn.Linear(hidden_dim, 3)
        self.answer_binary_head = nn.Linear(hidden_dim, 1)
        self.answer_attr_head = nn.Linear(hidden_dim, 15)
        self.answer_reg_head = MLP(hidden_dim, hidden_dim, 20, 3)

    def __call__(self, hs):
        out = {}
        answer_embeds = hs[0, :, -4:]
        hs = hs[:, :, :-4]
        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
        out["pred_answer_binary"] = self.answer_binary_head(answer_embeds[:, 1]).squeeze(-1)
        out["pred_answer_reg"] = self.answer_reg_head(answer_embeds[:, 2])
        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
        return out, hs


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    assert not args.masks or args.mask_model != "none"

    qa_dataset = validate_qa_dataset_choice(args)
    criterion, contrastive_criterion, qa_criterion = build_criterion(args, qa_dataset)
    weight_dict = build_weight_dict(args, qa_dataset)
    
    backbone = build_backbone(args)
    text_encoder = build_text_encoder(args)
    transformer = build_transformer(args)
    pos_encode = build_position_encoding(args)

    model = MDETR(
        backbone,
        text_encoder,
        transformer,
        pos_encode=pos_encode,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        qa_dataset=qa_dataset,
        split_qa_heads=args.split_qa_heads,
        predict_final=args.predict_final,
    )
    if args.mask_model != "none":
        model = DETRsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None))
    return model, criterion, contrastive_criterion, qa_criterion, weight_dict


def validate_qa_dataset_choice(args):
    qa_dataset = None
    if args.do_qa:
        assert not (
            ("clevr" in args.combine_datasets or "clevr_question" in args.combine_datasets)
            and "gqa" in args.combine_datasets
        ), "training GQA and CLEVR simultaneously is not supported"
        assert (
            "clevr_question" in args.combine_datasets
            or "clevr" in args.combine_datasets
            or "gqa" in args.combine_datasets
        ), "Question answering require either gqa or clevr dataset"
        qa_dataset = "gqa" if "gqa" in args.combine_datasets else "clevr"
    return qa_dataset


def build_weight_dict(args, qa_dataset):
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef}
    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.predict_final:
        weight_dict["loss_isfinal"] = 1

    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.do_qa:
        if args.split_qa_heads:
            weight_dict["loss_answer_type"] = 1 * args.qa_loss_coef
            if qa_dataset == "gqa":
                weight_dict["loss_answer_cat"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_attr"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_rel"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_obj"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_global"] = 1 * args.qa_loss_coef
            else:
                weight_dict["loss_answer_binary"] = 1
                weight_dict["loss_answer_attr"] = 1
                weight_dict["loss_answer_reg"] = 1

        else:
            weight_dict["loss_answer_total"] = 1 * args.qa_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        weight_dict.update({
            f"{k}_{i}": v 
            for i in range(args.dec_layers - 1)
            for k, v in weight_dict.items()
        })
    return weight_dict