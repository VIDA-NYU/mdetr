# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Dataset and evaluator for CLEVR-Ref+"""
import copy
from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data
# from transformers import RobertaTokenizerFast

import util.dist as dist
from datasets.clevr import make_clevr_transforms
from util.box_ops import generalized_box_iou

from .coco import ModulatedDetection, make_coco_transforms
from ..models.text_encoder import build_tokenizer


class ClevrRefDetection(ModulatedDetection):
    pass


class ClevrRefEvaluator(object):
    def __init__(self, refexp_gt, iou_types, k=(1, 5, 10), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            dataset2score = {
                "clevrref": {k: 0.0 for k in self.k},
            }
            dataset2count = {"clevrref": 0.0}
            for image_id in self.img_ids:
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                if len(ann_ids) != 1:
                    continue
                img_info = self.refexp_gt.loadImgs(image_id)[0]

                target = self.refexp_gt.loadAnns(ann_ids[0])
                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores_refexp"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                target_bbox = target[0]["bbox"]
                converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                giou = generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
                for k in self.k:
                    if max(giou[:k]) >= self.thresh_iou:
                        dataset2score["clevrref"][k] += 1.0
                dataset2count["clevrref"] += 1.0

            for key, value in dataset2score.items():
                for k in self.k:
                    try:
                        value[k] /= dataset2count[key]
                    except:
                        pass
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()])
                print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

            return results
        return None


def build(image_set, args):
    # tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    tokenizer = build_tokenizer(args)

    img_dir = Path(args.clevr_img_path) / f"{image_set}"
    ann_file = Path(args.clevr_ann_path) / f"{image_set}.json"

    dataset = ClevrRefDetection(
        img_dir,
        ann_file,
        transforms=make_clevr_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
    )
    return dataset
