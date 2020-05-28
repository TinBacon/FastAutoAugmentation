# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
import glob
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

from . import detection_utils as utils
from . import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True, policies=None):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None
        # Tin
        # fast_autoaug 0
        # self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        self.tfm_gens = utils.build_transform_gen(cfg, is_train, policies)
        self.autoaug = cfg.AUTOAUG.NUM_POLICY>0
        # Bacon

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.matting        = cfg.MODEL.RESNETS.MATTING
        self.input_size     = cfg.INPUT.MIN_SIZE_TRAIN[0]
        self.binary         = cfg.INPUT.BINARY
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        # Tin
        # fast_autoaug 1
        if not self.is_train and not self.autoaug:
        # Bacon
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            a = dataset_dict["sem_seg_file_name"]
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
                # Tin
                # matting 0
                sem_seg_gt = sem_seg_gt.copy()
                if sem_seg_gt.max() == 0:
                    assert 0, "label max 0"
                if self.matting:
                    # the matting datasets' labels are not unification
                    if sem_seg_gt.max() == 1:
                        sem_seg_gt = sem_seg_gt * 255
                # sem_fpn 0
                if self.binary:
                    sem_seg_gt[sem_seg_gt==255] = 1
                    sem_seg_gt = sem_seg_gt[:, :, 0].copy()
                # Bacon
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            # Tin
            # matting 1
            if self.matting or self.binary:
                sem_seg_gt[sem_seg_gt==255] = -1
                sem_seg_gt[sem_seg_gt>0] = 1
                sem_seg_gt[sem_seg_gt==-1] = 255
            # Bacon
            dataset_dict["sem_seg"] = sem_seg_gt

            if not self.is_train:
                import cv2
                cv2.imwrite("/home/pgding/project/LIP/detectron2_bdd/tools/output/tmp/"+a.split("/")[-1].replace(".png", "_orig.jpg"), dataset_dict["image"].numpy().transpose(1, 2, 0))
            # if self.is_train:
            #     s = sem_seg_gt.numpy().copy()
            #     s[s==1] = 100
            #     cv2.imwrite("/home/pgding/project/LIP/detectron2_bdd/tools/output/tmp/"+a.split("/")[-1].replace(".png", "_train_mp.jpg"), s)
        # Tin
        # matting 2
        if self.is_train and self.matting:

            img = self.merge_bg(dataset_dict["image"], dataset_dict["sem_seg"], dataset_dict["bg_file_dir"])
            dataset_dict["image"] = img
        # Bacon
        return dataset_dict

    # Tin
    # matting 3
    def merge_bg(self, img, label, bg_dir):
        
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # merge bg
        random_bg = np.random.choice([0, 1, 2])
        list_sample_bg = glob.glob(bg_dir+'/*.jpg') + glob.glob(bg_dir+'/*/*.jpg') + glob.glob(bg_dir+'/*/*/*.jpg')  + glob.glob(bg_dir+'/*/*/*/*.jpg') 
        
        # 0: original 
        # 1: list_sample_bg
        # 2: pure bg
        if random_bg == 1:
            bg_path = np.random.choice(list_sample_bg)
            bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            bg = cv2.resize(bg, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            # bg = bg.astype(np.float32) # [:, :, ::-1] # RGB to BGR!!!
        elif random_bg == 2:
            # Generate Bg
            bg = np.random.randint(255, size=3)
            bg = bg.reshape(1,1,3)
            bg = np.repeat(bg, self.input_size, axis=0)
            bg = np.repeat(bg, self.input_size, axis=1)
            bg = bg.astype(np.uint8)
            # Augmentation
            bg_seq = iaa.Sequential(
                [
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((1, 6),
                        [
                            sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                            # search either for all edges or for directed edges,
                            # blend the result with the original image using a blobby mask
                            iaa.SimplexNoiseAlpha(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                            ])),
                            iaa.OneOf([
                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                                iaa.AdditiveLaplaceNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                                iaa.AdditivePoissonNoise(lam=(0.0, 4.0), per_channel=0.5)
                            ]),
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.03, 0.2), size_percent=(0.02, 0.05), per_channel=0.2),
                            ]),
                            iaa.Invert(0.1, per_channel=True), # invert color channels
                            iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                            iaa.OneOf([
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-4, 0),
                                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                    second=iaa.ContrastNormalization((0.5, 2.0))
                                )
                            ]),
                            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                            iaa.Add((-25, 25), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                            iaa.OneOf([
                                iaa.ImpulseNoise((0.01, 0.1)),
                                iaa.SaltAndPepper((0.01, 0.1), per_channel=0.2),
                            ]),
                            iaa.JpegCompression(),

                        ],
                        random_order=True
                    ),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        iaa.JpegCompression(),
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    ]),
                ],
                random_order=False
            )
            bg = bg_seq.augment_image(bg)

        if random_bg >= 1:
            bg = torch.as_tensor(bg.astype(np.float32))
            bg = torch.transpose(torch.transpose(bg, 1, 2), 0, 1)
            img = img * label + bg * (1 - label)
        
        return img
    # Bacon