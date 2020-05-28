# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .register_coco import register_coco_instances, register_coco_panoptic_separated
from .lvis import register_lvis_instances, get_lvis_instances_meta
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .bdd import load_bdd_instances, load_bdd_semantic
from .pascal_voc import register_pascal_voc
from .builtin_meta import _get_builtin_metadata

# Tin
from .LIP import load_LIP_semantic
from .TJ import load_TJ0607_semantic, load_TJ0630_semantic
from .MATTING import load_MATTING_semantic
from .IDDATA import load_IDDATA_semantic
from .HUMAN_HALF import load_HUMAN_HALF_semantic
from .FULLBODY import load_FULLBODY_semantic
from .FIND_HUMAN import load_FIND_HUMAN_semantic, load_FIND_HUMAN_MATTING_semantic
from .MSCOCO import load_MSCOCO_semantic
# Bacon

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/val2017", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/test2017", "lvis/lvis_v0.5_image_info_test.json"),
    }
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
}


def register_all_cityscapes(root="datasets"):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="sem_seg", **meta
        )


_RAW_BDD_SPLITS = {
    "cityscapes_bdd_{task}_train": ("bdd/train/images", "bdd/train/label_ids"),
    "cityscapes_bdd_{task}_val": ("bdd/val/images", "bdd/val/label_ids"),
}


def register_all_bdd(root="datasets"):
    for key, (image_dir, gt_dir) in _RAW_BDD_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        # inst_key = key.format(task="instance_seg")
        # DatasetCatalog.register(
        #     inst_key,
        #     lambda x=image_dir, y=gt_dir: load_bdd_instances(
        #         x, y, from_json=True, to_polygons=True
        #     ),
        # )
        # MetadataCatalog.get(inst_key).set(
        #     image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
        # )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_bdd_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="sem_seg", **meta
        )


# Tin
# Matting 0
bg_dir = "/data/zhubingke/Alpha_Aug/bg_images"

# ==== 1. Predefined splits for LIP ===========
_RAW_LIP_SPLITS = {
    "LIP_train":("train_image", "train_segmentation"),
    "LIP_autoaug_train":("autoaug_train_image", "autoaug_train_segmentation"),
    "LIP_autoaug_test":("autoaug_test_image", "autoaug_test_segmentation"),
    "LIP_smoke_train":("smoke_train", "smoke_train_seg"),
    "LIP_test":("test_image", "INVALID"),
}

def register_all_LIP(root="/data/zhubingke/LIP-Portrait/"):
    for key, (image_dir, gt_dir) in _RAW_LIP_SPLITS.items():
        meta = _get_builtin_metadata("LIP")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        
        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_LIP_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)

# ==== 2. Predefined splits for MATTING ===========
_RAW_HUMAN_HALF_MATTING_SPLITS = {
    "HUMAN_HALF_MATTING_train":("image", "alpha"),
    # "HUMAN_HALF_MATTING_test":("val_image", "val_segmentation"),
}

def register_all_HUMAN_HALF_MATTING(root="/data/zhubingke/matting_human_half"):
    for key, (image_dir, gt_dir) in _RAW_HUMAN_HALF_MATTING_SPLITS.items():
        meta = _get_builtin_metadata("HUMAN_HALF_MATTING")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_HUMAN_HALF_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 3. Predefined splits for FINDHUMAN2000 ===========
_RAW_FIND_HUMAN_2000_SPLITS = {
    "FIND_HUMAN_2000_train":("Images", "Labels20170424/Labels"),
    # "FIND_HUMAN_2000_test":("test", "val_segmentation"),
}

def register_all_FIND_HUMAN_2000(root="/data/zhubingke/findhuman/2000"):
    for key, (image_dir, gt_dir) in _RAW_FIND_HUMAN_2000_SPLITS.items():
        meta = _get_builtin_metadata("FIND_HUMAN_2000")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_FIND_HUMAN_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 4. Predefined splits for FINDHUMAN2000MATTING ===========
_RAW_FIND_HUMAN_2000_MATTING_SPLITS = {
    "FIND_HUMAN_2000_MATTING_train":("image", "label"),
    # "FIND_HUMAN_2000_MATTING_test":("val_image", "val_segmentation"),
}

def register_all_FIND_HUMAN_2000_MATTING(root="/data/zhubingke/findhuman/2000/matting_data/training_new"):
    for key, (image_dir, gt_dir) in _RAW_FIND_HUMAN_2000_MATTING_SPLITS.items():
        meta = _get_builtin_metadata("FIND_HUMAN_2000_MATTING")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_FIND_HUMAN_MATTING_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 5. Predefined splits for FULLBODYPRETRAINED ===========
_RAW_FULLBODY_PRETRAINED_SPLITS = {
    "FULLBODY_PRETRAINED_train":("pretrain_img", "pretrain_alpha"),
    # "FULLBODY_PRETRAINED_test":("val_image", "val_segmentation"),
}

def register_all_FULLBODY_PRETRAINED(root="/data/zhubingke/Alpha_Aug/Full_Body_MultiHuman_Data_20191119"):
    for key, (image_dir, gt_dir) in _RAW_FULLBODY_PRETRAINED_SPLITS.items():
        meta = _get_builtin_metadata("FULLBODY")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_FULLBODY_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 6. Predefined splits for FULLBODYPRETRAINED ===========
_RAW_FULLBODY_SPLITS = {
    "FULLBODY_train":("img", "alpha"),
    # "FULLBODY_test":("val_image", "val_segmentation"),
}

def register_all_FULLBODY(root="/data/zhubingke/Alpha_Aug/Full_Body_MultiHuman_Data_20191119"):
    for key, (image_dir, gt_dir) in _RAW_FULLBODY_SPLITS.items():
        meta = _get_builtin_metadata("FULLBODY")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_FULLBODY_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 7. Predefined splits for IDData_20190604 ===========
_RAW_IDDATA_20190604 = {
    "IDDATA_20190604_train":("img", "alpha"),
    # "IDDATA_20190604_test":("val_image", "val_segmentation"),
}

def register_all_IDDATA_20190604(root="/data/zhubingke/Alpha_Aug/IDData_20190604"):
    for key, (image_dir, gt_dir) in _RAW_IDDATA_20190604.items():
        meta = _get_builtin_metadata("IDDATA")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_IDDATA_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 8. Predefined splits for matting_190619 ===========
_RAW_MATTING_190619 = {
    "MATTING_190619_train":("img", "alpha"),
    # "MATTING_190619_test":("val_image", "val_segmentation"),
}

def register_all_MATTING_190619(root="/data/zhubingke/Alpha_Aug/matting_190619"):
    for key, (image_dir, gt_dir) in _RAW_MATTING_190619.items():
        meta = _get_builtin_metadata("MATTING")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_MATTING_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 9. Predefined splits for matting_190705 ===========
_RAW_MATTING_190705 = {
    "MATTING_190705_train":("img", "alpha"),
    # "MATTING_190705_test":("val_image", "val_segmentation"),
}

def register_all_MATTING_190705(root="/data/zhubingke/Alpha_Aug/matting_190705"):
    for key, (image_dir, gt_dir) in _RAW_MATTING_190705.items():
        meta = _get_builtin_metadata("MATTING")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_MATTING_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 10. Predefined splits for matting_190909 ===========
_RAW_MATTING_190909 = {
    "MATTING_190909_train":("img", "alpha"),
    # "MATTING_190909_test":("val_image", "val_segmentation"),
}

def register_all_MATTING_190909(root="/data/zhubingke/Alpha_Aug/matting_190909"):
    for key, (image_dir, gt_dir) in _RAW_MATTING_190909.items():
        meta = _get_builtin_metadata("MATTING")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_MATTING_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 11. Predefined splits for matting_190912 ===========
_RAW_MATTING_190912 = {
    "MATTING_190912_train":("img", "alpha"),
    # "MATTING_190912_test":("val_image", "val_segmentation"),
}

def register_all_MATTING_190912(root="/data/zhubingke/Alpha_Aug/matting_190912"):
    for key, (image_dir, gt_dir) in _RAW_MATTING_190912.items():
        meta = _get_builtin_metadata("MATTING")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_MATTING_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 12. Predefined splits for TJ0630 ===========
_RAW_TJ0630 = {
    "TJ0630_train":("rgba", "alph"),
    # "TJ0630_test":("val_image", "val_segmentation"),
}

def register_all_TJ0630(root="/data/zhubingke/tj0630"):
    for key, (image_dir, gt_dir) in _RAW_TJ0630.items():
        meta = _get_builtin_metadata("TJ")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_TJ0630_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 13. Predefined splits for TJ0607 ===========
_RAW_TJ0607 = {
    "TJ0607_train":("rgb", "alpha"),
    # "TJ0607_test":("val_image", "val_segmentation"),
}

def register_all_TJ0607(root="/data/zhubingke/tj0607"):
    for key, (image_dir, gt_dir) in _RAW_TJ0607.items():
        meta = _get_builtin_metadata("TJ")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_TJ0607_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)


# ==== 14. Predefined splits for MSCOCO ===========
_RAW_MSCOCOHUMANPLUS = {
    "MSCOCOHUMANPLUS_train":("img", "mask"),
    # "MSCOCOHUMANPLUS_test":("val_image", "val_segmentation"),
}

def register_all_MSCOCOHUMANPLUS(root="/data/zhubingke/MSCOCOHumanPlus/Supervisely_Join"):
    for key, (image_dir, gt_dir) in _RAW_MSCOCOHUMANPLUS.items():
        meta = _get_builtin_metadata("MSCOCO")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=bg_dir: load_MSCOCO_semantic(x, y, z))
        MetadataCatalog.get(key).set(iamge_dir=image_dir, gt_dir=gt_dir, evaluator_type="MATTING", **meta)
# Bacon

# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
register_all_coco()
register_all_lvis()
register_all_cityscapes()
register_all_bdd()
register_all_pascal_voc()
# Tin
# matting 1
register_all_LIP()
register_all_HUMAN_HALF_MATTING()
register_all_FIND_HUMAN_2000()
register_all_FIND_HUMAN_2000_MATTING()
register_all_FULLBODY_PRETRAINED()
register_all_FULLBODY()
register_all_IDDATA_20190604()
register_all_MATTING_190619()
register_all_MATTING_190705()
register_all_MATTING_190909()
register_all_MATTING_190912()
register_all_TJ0630()
register_all_TJ0607()
register_all_MSCOCOHUMANPLUS()
# Bacon