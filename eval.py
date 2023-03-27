#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Evaluation script.
"""

import os, pdb
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from hos.config import add_hos_config
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config

# register dataset
from detectron2.data import MetadataCatalog
from hos.data.datasets.epick import register_epick_instances
from hos.data.hos_datasetmapper import HOSMapper
from hos.evaluation.epick_evaluation import EPICKEvaluator


version = 'datasets/epick_visor_coco_hos'
register_epick_instances("epick_visor_2022_val_hos", {}, f"{version}/annotations/val.json", f"{version}/val")
MetadataCatalog.get("epick_visor_2022_val_hos").thing_classes = ["hand", "object"]

version = 'datasets/epick_visor_coco_contact'
register_epick_instances("epick_visor_2022_val_contact", {}, f"{version}/annotations/val.json", f"{version}/val")
MetadataCatalog.get("epick_visor_2022_val_contact").thing_classes = ["not_incontact", 'incontact']

version = 'datasets/epick_visor_coco_handside'
register_epick_instances("epick_visor_2022_val_handside", {}, f"{version}/annotations/val.json", f"{version}/val")
MetadataCatalog.get("epick_visor_2022_val_handside").thing_classes = ["left", "right"]

version = 'datasets/epick_visor_coco_active'
register_epick_instances("epick_visor_2022_val_active", {}, f"{version}/annotations/val.json", f"{version}/val")
MetadataCatalog.get("epick_visor_2022_val_active").thing_classes = ["hand", "object"]

version = 'datasets/epick_visor_coco_combineHO'
register_epick_instances("epick_visor_2022_val_combineHO", {}, f"{version}/annotations/val.json", f"{version}/val")
MetadataCatalog.get("epick_visor_2022_val_combineHO").thing_classes = ["combineHandObj"]


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    return augs


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        
        if evaluator_type == "coco":
            evaluator_list = [
                # choose the task you want to evaluate below, and indicate the corresponding dataset used in the config file
                
                EPICKEvaluator('epick_visor_2022_val_hos', output_dir=output_folder, eval_task='obj_box'),
                # EPICKEvaluator('epick_visor_2022_val_handside', output_dir=output_folder, eval_task='handside'),
                # EPICKEvaluator('epick_visor_2022_val_contact', output_dir=output_folder, eval_task='contact'),
                # EPICKEvaluator('epick_visor_2022_val_combineHO', output_dir=output_folder, eval_task='combineHO'),
                # COCOEvaluator('epick_visor_2022_val_active', output_dir=output_folder), 
        
                ]
            return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        mapper = HOSMapper(cfg)
        print(f'**dataset mapper used: {mapper}, {cfg.MODEL.META_ARCHITECTURE}')
        # pdb.set_trace()
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print(f'here are the configs:\n {cfg}')
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    args.dist_url = f"tcp://127.0.0.1:60111"
    print("Command Line Args:", args)

    # run
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )