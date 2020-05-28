# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File: transform.py

import numpy as np
from fvcore.transforms.transform import HFlipTransform, NoOpTransform, Transform
from PIL import Image
import cv2
import random
import logging

from detectron2.utils.autoaug import *

__all__ = ["ExtentTransform", "ResizeTransform", "RotateTransform", "MattingTransform", "CutoutTransform", "AutoAugTransform"]


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside
    the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

# Tin
# matting 0
class RotateTransform(Transform):
    def __init__(self, h, w, box0, box1):
        super().__init__()
        
        self.rotate     = np.random.random()<0.5
        self.borderMode = cv2.BORDER_CONSTANT
        self.scale      = (w, h)
        self.mat        = cv2.getPerspectiveTransform(box0, box1)
    
    def apply_image(self, image, interp=cv2.INTER_LINEAR, borderValue=(0, 0, 0)):
        if self.rotate:
            image = cv2.warpPerspective(image, self.mat, self.scale, flags=interp, borderMode=self.borderMode, borderValue=borderValue)
        return image

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST, borderValue=(255))
        return segmentation

    def apply_coords(self, coords):
        pass

class MattingTransform(Transform):

    def __init__(self, input_size):
        super().__init__()
        
        self.input_size = input_size
    
    def apply_image(self, image, interp=cv2.INTER_LINEAR):
        
        return cv2.resize(image, (self.input_size, self.input_size), interpolation=interp)

    def apply_segmentation(self, segmentation):
        
        return self.apply_image(segmentation, interp=cv2.INTER_NEAREST)

    def apply_coords(self, coords):
        pass

# fast autoaug 0
class CutoutTransform(Transform):

    def __init__(self, length):
        super().__init__()
        
        self.length = length
        self.seed = [random.random(), random.random()]
        self.fill = 255.
    
    def apply_image(self, img):

        if not self.length:
            return img

        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = h * self.seed[0]
        x = w * self.seed[1]

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = self.fill
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

    def apply_segmentation(self, segmentation):
        
        return self.apply_image(segmentation)

    def apply_coords(self, coords):
        pass

class AutoAugTransform(Transform):

    def __init__(self, policies):
        super().__init__()
        
        self.policy       = random.choice(policies)
        # random augmentation
        self.aug_seed     = [random.random() for _ in self.policy]
        # some augments are not fit with labels
        self.not_seg      = ["AutoContrast", "Invert", "Equalize", "Solarize", "Posterize", "Contrast", "Color", "Brightness", "Sharpness"]
        # some augments can neg the magnitude
        self.mirror       = ["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXAbs", "TranslateYAbs", "Rotate"]
        self.mirror_seed  = [random.random() for _ in self.policy]

    
    def apply_image(self, img, seg=False):

        logger = logging.getLogger("detectron2.trainer")

        img = Image.fromarray(img)
        for i, (name, pr, level) in enumerate(self.policy):
            
            # some augments are not fit with labels
            if self.aug_seed[i] > pr or (seg and name in self.not_seg):
                continue
            
            augment_fn, low, high = augment_dict[name]
            magnitude = level * (high - low) + low

            # random mirror 
            if name in self.mirror and self.mirror_seed[i] > 0.5:
                magnitude = -magnitude

            # data augment
            img = augment_fn(img, magnitude)

            if len(np.unique(img))==1:
                logger.info("transform {} make the image wrong".format(name))

        return np.array(img)

    def apply_segmentation(self, segmentation):
        
        return self.apply_image(segmentation, True)

    def apply_coords(self, coords):
        pass

# Bacon

def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
NoOpTransform.register_type("rotated_box", lambda t, x: x)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)
