# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

# from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
# from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

# from ..backbone.resnet import BottleneckBlock, ResNet
# from ..matcher import Matcher
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
# from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
# from ..sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
# from .fast_rcnn import FastRCNNOutputLayers
# from .keypoint_head import build_keypoint_head
# from .mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import (
    StandardROIHeads,
    select_foreground_proposals,
)

from hos.modeling.roi_heads.predictor import HOSFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class HOSROIHeads(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
    #     self._init_box_head(cfg, input_shape)


    # def _init_box_head(self, cfg, input_shape):
        # fmt: off
        self.in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = HOSFastRCNNOutputLayers(cfg, self.box_head.output_shape)
        
    
    # @torch.no_grad()
    # def label_and_sample_proposals(
    #     self, proposals: List[Instances], targets: List[Instances]
    # ):# -> List[Instances]:
    #     """
    #     Prepare some proposals to be used to train the ROI heads.
    #     It performs box matching between `proposals` and `targets`, and assigns
    #     training labels to the proposals.
    #     It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
    #     boxes, with a fraction of positives that is no larger than
    #     ``self.positive_fraction``.

    #     Args:
    #         See :meth:`ROIHeads.forward`

    #     Returns:
    #         list[Instances]:
    #             length `N` list of `Instances`s containing the proposals
    #             sampled for training. Each `Instances` has the following fields:

    #             - proposal_boxes: the proposal boxes
    #             - gt_boxes: the ground-truth box that the proposal is assigned to
    #               (this is only meaningful if the proposal has a label > 0; if label = 0
    #               then the ground-truth box is random)

    #             Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
    #     """
    #     # Augment proposals with ground-truth boxes.
    #     # In the case of learned proposals (e.g., RPN), when training starts
    #     # the proposals will be low quality due to random initialization.
    #     # It's possible that none of these initial
    #     # proposals have high enough overlap with the gt objects to be used
    #     # as positive examples for the second stage components (box head,
    #     # cls head, mask head). Adding the gt boxes to the set of proposals
    #     # ensures that the second stage components will have some positive
    #     # examples from the start of training. For RPN, this augmentation improves
    #     # convergence and empirically improves box AP on COCO by about 0.5
    #     # points (under one tested configuration).
    #     if self.proposal_append_gt:
    #         proposals = add_ground_truth_to_proposals(targets, proposals)

    #     proposals_with_gt = []

    #     num_fg_samples = []
    #     num_bg_samples = []
    #     for proposals_per_image, targets_per_image in zip(proposals, targets):
    #         has_gt = len(targets_per_image) > 0
    #         match_quality_matrix = pairwise_iou(
    #             targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
    #         )
    #         matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
    #         sampled_idxs, gt_classes = self._sample_proposals(
    #             matched_idxs, matched_labels, targets_per_image.gt_classes
    #         )

    #         # Set target attributes of the sampled proposals:
    #         proposals_per_image = proposals_per_image[sampled_idxs]
    #         proposals_per_image.gt_classes = gt_classes

    #         if has_gt:
    #             sampled_targets = matched_idxs[sampled_idxs]
    #             # We index all the attributes of targets that start with "gt_"
    #             # and have not been added to proposals yet (="gt_classes").
    #             # NOTE: here the indexing waste some compute, because heads
    #             # like masks, keypoints, etc, will filter the proposals again,
    #             # (by foreground/background, or number of keypoints in the image, etc)
    #             # so we essentially index the data twice.
    #             for (trg_name, trg_value) in targets_per_image.get_fields().items():
    #                 if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
    #                     proposals_per_image.set(trg_name, trg_value[sampled_targets])
    #         # If no GT is given in the image, we don't know what a dummy gt value can be.
    #         # Therefore the returned proposals won't have any gt_* fields, except for a
    #         # gt_classes full of background label.

    #         num_bg_samples.append((gt_classes == self.num_classes).sum().item())
    #         num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
    #         proposals_with_gt.append(proposals_per_image)

    #     # Log the number of fg/bg samples that are selected for training ROI heads
    #     storage = get_event_storage()
    #     storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
    #     storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

    #     return proposals_with_gt




    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
            """
            Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
                the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

            Args:
                features (dict[str, Tensor]): mapping from feature map names to tensor.
                    Same as in :meth:`ROIHeads.forward`.
                proposals (list[Instances]): the per-image object proposals with
                    their matching ground truth.
                    Each has fields "proposal_boxes", and "objectness_logits",
                    "gt_classes", "gt_boxes".

            Returns:
                In training, a dict of losses.
                In inference, a list of `Instances`, the predicted instances.
            """
            features = [features[f] for f in self.box_in_features]
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features)
            predictions = self.box_predictor(box_features)
            del box_features

            if self.training:
                losses = self.box_predictor.losses(predictions, proposals)
                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                return losses
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                return pred_instances
            
            
    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
        ):
            """
            See :class:`ROIHeads.forward`.
            """
            # print(f'** features = {features}')
            # print(f'** proposals = {proposals}')
            # print(f'** targets = {targets}')
            del images
            if self.training:
                assert targets, "'targets' argument is required during training"
                proposals = self.label_and_sample_proposals(proposals, targets)
            del targets

            if self.training:
                losses = self._forward_box(features, proposals)
                # Usually the original proposals used by the box head are used by the mask, keypoint
                # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
                # predicted by the box head.
                losses.update(self._forward_mask(features, proposals))
                losses.update(self._forward_keypoint(features, proposals))
                return proposals, losses
            else:
                pred_instances = self._forward_box(features, proposals)
                # During inference cascaded prediction is used: the mask and keypoints heads are only
                # applied to the top scoring box detections.
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
                return pred_instances, {}