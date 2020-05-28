# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from detectron2.structures import ImageList

from ..backbone import build_backbone
from ..postprocessing import sem_seg_postprocess
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from .semantic_seg import build_sem_seg_head

__all__ = ["SemanticFPN"]


@META_ARCH_REGISTRY.register()
class SemanticFPN(nn.Module):
    """
    Main class for Semantic FPN architectures transformed from Panoptic FPN.
    """

    def __init__(self, cfg):

        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: each dict is the results for one image. The dict
                contains the following keys:
                "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "sem_seg" not in batched_inputs[0]:
            gt_sem_segs_ = None
            gt_sem_segs = [x["image"][0].to(self.device) for x in batched_inputs]
        else:
            gt_sem_segs = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_segs_ = ImageList.from_tensors(
                gt_sem_segs, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_segs_)

        if self.training:

            sem_seg_result, input_per_image, image_size, gt_sem_seg, name = sem_seg_results[0], batched_inputs[0], images.image_sizes[0], gt_sem_segs[0], batched_inputs[0]["file_name"]    

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_result = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            sem_seg_result = torch.max(sem_seg_result.cpu(), dim=0)[1].data.numpy()*255

            # post process ground truth
            gt_sem_seg = gt_sem_seg[: image_size[0], : image_size[1]].expand(1, 1, -1, -1).float()
            gt_sem_seg = F.interpolate(gt_sem_seg, size=(height, width), mode="nearest")[0][0]
            gt_sem_seg[gt_sem_seg==255] = -1
            gt_sem_seg[gt_sem_seg>0] = 100
            gt_sem_seg[gt_sem_seg==-1] = 255
            gt_sem_seg = gt_sem_seg.int().cpu().numpy()

            return sem_seg_losses, [name, gt_sem_seg, sem_seg_result]
        else:
            processed_results = []
            for sem_seg_result, input_per_image, image_size, gt_sem_seg in zip(sem_seg_results, batched_inputs, images.image_sizes, gt_sem_segs):           
                
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_result = sem_seg_postprocess(sem_seg_result, image_size, height, width)

                # post process ground truth
                gt_sem_seg = gt_sem_seg[: image_size[0], : image_size[1]].expand(1, 1, -1, -1).float()
                gt_sem_seg = F.interpolate(gt_sem_seg, size=(height, width), mode="nearest")[0][0]
                gt_sem_seg[gt_sem_seg==255] = -1
                gt_sem_seg[gt_sem_seg>0] = 1
                gt_sem_seg[gt_sem_seg==-1] = 255

                processed_results.append({"sem_seg": sem_seg_result, "gt_semseg":gt_sem_seg, "loss":sem_seg_losses["loss_sem_seg"]})

            return processed_results