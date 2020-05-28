# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .cityscapes import load_cityscapes_instances
from .bdd import load_bdd_instances
from .coco import load_coco_json, load_sem_seg
from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .register_coco import register_coco_instances, register_coco_panoptic_separated
from . import builtin  # ensure the builtin datasets are registered

# Tin
from .LIP import load_LIP_semantic
from .TJ import load_TJ0607_semantic, load_TJ0630_semantic
from .MATTING import load_MATTING_semantic
from .IDDATA import load_IDDATA_semantic
from .HUMAN_HALF import load_HUMAN_HALF_semantic
from .FULLBODY import load_FULLBODY_semantic
from .FIND_HUMAN import load_FIND_HUMAN_semantic, load_FIND_HUMAN_MATTING_semantic
# Bacon

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
