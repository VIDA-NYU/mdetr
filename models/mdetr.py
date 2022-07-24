# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import util.dist as dist
from util import box_ops
from util.metrics import accuracy
from util.misc import NestedTensor, interpolate

from .backbone import build_backbone
from .matcher import build_matcher
from .postprocessors import build_postprocessors
from .segmentation import DETRsegm, dice_loss, sigmoid_focal_loss
from .transformer import build_transformer
from .criterion import SetCriterion, ContrastiveCriterion, QACriterionGQA, QACriterionClevr


class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
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
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if qa_dataset is not None:
            nb_heads = 6 if qa_dataset == "gqa" else 4
            self.qa_embed = nn.Embedding(nb_heads if split_qa_heads else 1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
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
            if split_qa_heads:
                self.answer_type_head = nn.Linear(hidden_dim, 5)
                # TODO: make this more general
                if qa_dataset == "gqa":
                    self.answer_rel_head = nn.Linear(hidden_dim, 1594)
                    self.answer_obj_head = nn.Linear(hidden_dim, 3)
                    self.answer_global_head = nn.Linear(hidden_dim, 111)
                    self.answer_attr_head = nn.Linear(hidden_dim, 403)
                    self.answer_cat_head = nn.Linear(hidden_dim, 678)
                elif qa_dataset == "clevr":
                    self.answer_type_head = nn.Linear(hidden_dim, 3)
                    self.answer_binary_head = nn.Linear(hidden_dim, 1)
                    self.answer_attr_head = nn.Linear(hidden_dim, 15)
                    self.answer_reg_head = MLP(hidden_dim, hidden_dim, 20, 3)
                else:
                    assert False, f"Invalid qa dataset {qa_dataset}"
            else:
                # TODO: make this more general
                assert qa_dataset == "gqa", "Clevr QA is not supported with unified head"
                self.answer_head = nn.Linear(hidden_dim, 1853)

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
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            query_embed = self.query_embed.weight
            if self.qa_dataset is not None:
                query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)
            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                query_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            )

            if self.contrastive_loss:
                memory_cache["text_pooled_op"] = self.contrastive_projection_text(memory_cache["text_pooled_op"])
                memory_cache["img_pooled_op"] = self.contrastive_projection_image(memory_cache["img_pooled_op"])

            return memory_cache

        else:
            assert memory_cache is not None
            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            )
            out = {}
            if self.qa_dataset is not None:
                if self.split_qa_heads:
                    if self.qa_dataset == "gqa":
                        answer_embeds = hs[0, :, -6:]
                        hs = hs[:, :, :-6]
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_obj"] = self.answer_obj_head(answer_embeds[:, 1])
                        out["pred_answer_rel"] = self.answer_rel_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                        out["pred_answer_cat"] = self.answer_cat_head(answer_embeds[:, 4])
                        out["pred_answer_global"] = self.answer_global_head(answer_embeds[:, 5])
                    elif self.qa_dataset == "clevr":
                        answer_embeds = hs[0, :, -4:]
                        hs = hs[:, :, :-4]
                        out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                        out["pred_answer_binary"] = self.answer_binary_head(answer_embeds[:, 1]).squeeze(-1)
                        out["pred_answer_reg"] = self.answer_reg_head(answer_embeds[:, 2])
                        out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                    else:
                        assert False, f"Invalid qa dataset {self.qa_dataset}"

                else:
                    answer_embeds = hs[0, :, -1]
                    hs = hs[:, :, :-1]
                    out["pred_answer"] = self.answer_head(answer_embeds)

            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                }
            )
            outputs_isfinal = None
            if self.isfinal_embed is not None:
                outputs_isfinal = self.isfinal_embed(hs)
                out["pred_isfinal"] = outputs_isfinal[-1]
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )
            if self.aux_loss:
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
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
    num_classes = 255
    device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

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

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
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
            freeze_detr=(args.frozen_weights is not None),
        )
    matcher = build_matcher(args)
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
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    if args.predict_final:
        losses += ["isfinal"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = None
    if not args.no_detection:
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            eos_coef=args.eos_coef,
            losses=losses,
            temperature=args.temperature_NCE,
        )
        criterion.to(device)

    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE)
        contrastive_criterion.to(device)
    else:
        contrastive_criterion = None

    if args.do_qa:
        if qa_dataset == "gqa":
            qa_criterion = QACriterionGQA(split_qa_heads=args.split_qa_heads)
        elif qa_dataset == "clevr":
            qa_criterion = QACriterionClevr()
        else:
            assert False, f"Invalid qa dataset {qa_dataset}"
        qa_criterion.to(device)
    else:
        qa_criterion = None
    return model, criterion, contrastive_criterion, qa_criterion, weight_dict
