# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
from collections import defaultdict

fill = 255

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), fillcolor=fill)


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), fillcolor=fill)


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=fill)


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=fill)


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=fill)


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=fill)


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    return img.rotate(v, fillcolor=fill)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v, seed):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v, seed)


def CutoutAbs(img, v, seed):  # [0, 60] => percentage: [0, 0.2]

    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = w * seed[0]
    y0 = h * seed[1]

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    # color = (125, 123, 114)
    color = fill if len(img.split())<3 else (fill,fill,fill)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)

    return img


# def SamplePairing(imgs):  # [0, 0.4]
#     def f(img1, v):
#         i = np.random.choice(len(imgs))
#         img2 = PIL.Image.fromarray(imgs[i])
#         return PIL.Image.blend(img1, img2, v)

#     return f


def augment_list():  # 16 oeprations and their ranges

    return [
        (ShearX, -0.3, 0.3),            # 0                                         loss nearly dont down at 1 and flucuate AND miou normal
        (ShearY, -0.3, 0.3),            # 1                                         loss down fast at 0.3 AND miou ABnormal / loss down fast at 0.3 but fluctuate AND miou normal
        (TranslateX, -0.45, 0.45),      # 2                                         loss down fast at 0.3 but fluctuate AND miou ABnormal
        (TranslateY, -0.45, 0.45),      # 3                                         loss down fast at 0.3 AND miou normal
        (Rotate, -30, 30),              # 4                                         loss down fast at 0.3 AND miou normal
        (AutoContrast, 0, 1),           # 5  invalid for binary annotation image    loss down fast at 0.3 AND miou normal
        (Invert, 0, 1),                 # 6  invalid for binary annotation image    loss down fast at 0.3 AND miou normal
        (Equalize, 0, 1),               # 7  invalid for binary annotation image    
        (Solarize, 0, 256),             # 8  invalid for binary annotation image    
        (Posterize, 4, 8),              # 9  invalid for binary annotation image    
        (Contrast, 0.1, 1.9),           # 10 invalid for binary annotation image    
        (Color, 0.1, 1.9),              # 11 invalid for binary annotation image    
        (Brightness, 0.1, 1.9),         # 12 invalid for binary annotation image    
        (Sharpness, 0.1, 1.9),          # 13 invalid for binary annotation image    
        # (Cutout, 0, 0.2),               # 14                                      loss down fast at 0.3 AND miou normal
        # (SamplePairing(imgs), 0, 0.4),# 15 
    ]


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone