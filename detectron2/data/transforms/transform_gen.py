# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File: transformer.py

import inspect
import numpy as np
import pprint
import sys
from abc import ABCMeta, abstractmethod
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from PIL import Image
import cv2

from .transform import ExtentTransform, ResizeTransform, RotateTransform, MattingTransform, CutoutTransform, AutoAugTransform
__all__ = [
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "Resize",
    "ResizeShortestEdge",
    "TransformGen",
    "apply_transform_gens",
    # Tin
    "randomShiftScaleRotate",
    "Rotate90",
    "MattingResize",
    "Cutout",
    "AutoAug",
    # Bacon
]


def check_dtype(img):
    assert isinstance(img, np.ndarray), "[TransformGen] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


class TransformGen(metaclass=ABCMeta):
    """
    TransformGen takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


class RandomFlip(TransformGen):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(TransformGen):
    """ Resize image to a target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        
        return ResizeTransform(h, w, newh, neww, self.interp)


class RandomCrop(TransformGen):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomExtent(TransformGen):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        img_h, img_w = img.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(TransformGen):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(TransformGen):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(TransformGen):
    """
    Randomly transforms image saturation.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(TransformGen):
    """
    Randomly transforms image color using fixed PCA over ImageNet.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, img):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )

# Tin
# matting 0
class randomShiftScaleRotate(TransformGen):

    def __init__(self):

        self.shift_limit = (-0.01, 0.01)
        self.scale_limit = (-0.01, 0.01)
        self.rotate_limit = (-90.0, 90.0)
        self.aspect_limit = (-0.01, 0.01)
                
    def get_transform(self, image):

        height, width, _ = image.shape

        angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])
        scale = np.random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1])
        aspect = np.random.uniform(1 + self.aspect_limit[0], 1 + self.aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * width)
        dy = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)

        return RotateTransform(height, width, box0, box1)


class Rotate90(TransformGen):
    def __init__(self):
        pass

    def get_transform(self, image):

        height, width, _ = image.shape

        angle = 90.
        cc = np.math.cos(angle / 180 * np.math.pi) * 1.
        ss = np.math.sin(angle / 180 * np.math.pi) * 1.
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + 0., height / 2 + 0.])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
		
        return RotateTransform(height, width, box0, box1)


class MattingResize(TransformGen):

    def __init__(self, input_size):

        self.input_size = input_size

    def get_transform(self, image):

        return MattingTransform(self.input_size)


# fast_autoaug 0
class Cutout(TransformGen):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def get_transform(self, img):
        
        return CutoutTransform(self.length)


class AutoAug(TransformGen):
    
    def __init__(self, policies=None):

        if policies:
            self.policies = policies
        else:
            # initial policies
            self.policies = [[["ShearY", 0.14143816458479197, 0.513124791615952], ["Sharpness", 0.9290316227291179, 0.9788406212603302]], [["Color", 0.21502874228385338, 0.3698477943880306], ["TranslateY", 0.49865058747734736, 0.4352676987103321]], [["Brightness", 0.6603452126485386, 0.6990174510500261], ["Cutout", 0.7742953773992511, 0.8362550883640804]], [["Posterize", 0.5188375788270497, 0.9863648925446865], ["TranslateY", 0.8365230108655313, 0.6000972236440252]], [["ShearY", 0.9714994964711299, 0.2563663552809896], ["Equalize", 0.8987567223581153, 0.1181761775609772]], [["Sharpness", 0.14346409304565366, 0.5342189791746006], ["Sharpness", 0.1219714162835897, 0.44746801278319975]], [["TranslateX", 0.08089260772173967, 0.028011721602479833], ["TranslateX", 0.34767877352421406, 0.45131294688688794]], [["Brightness", 0.9191164585327378, 0.5143232242627864], ["Color", 0.9235247849934283, 0.30604586249462173]], [["Contrast", 0.4584173187505879, 0.40314219914942756], ["Rotate", 0.550289356406774, 0.38419022293237126]], [["Posterize", 0.37046156420799325, 0.052693291117634544], ["Cutout", 0.7597581409366909, 0.7535799791937421]], [["Color", 0.42583964114658746, 0.6776641859552079], ["ShearY", 0.2864805671096011, 0.07580175477739545]], [["Brightness", 0.5065952125552232, 0.5508640233704984], ["Brightness", 0.4760021616081475, 0.3544313318097987]], [["Posterize", 0.5169630851995185, 0.9466018906715961], ["Posterize", 0.5390336503396841, 0.1171015788193209]], [["Posterize", 0.41153170909576176, 0.7213063942615204], ["Rotate", 0.6232230424824348, 0.7291984098675746]], [["Color", 0.06704687234714028, 0.5278429246040438], ["Sharpness", 0.9146652195810183, 0.4581415618941407]], [["ShearX", 0.22404644446773492, 0.6508620171913467], ["Brightness", 0.06421961538672451, 0.06859528721039095]], [["Rotate", 0.29864103693134797, 0.5244313199644495], ["Sharpness", 0.4006161706584276, 0.5203708477368657]], [["AutoContrast", 0.5748186910788027, 0.8185482599354216], ["Posterize", 0.9571441684265188, 0.1921474117448481]], [["ShearY", 0.5214786760436251, 0.8375629059785009], ["Invert", 0.6872393349333636, 0.9307694335024579]], [["Contrast", 0.47219838080793364, 0.8228524484275648], ["TranslateY", 0.7435518856840543, 0.5888865560614439]], [["Posterize", 0.10773482839638836, 0.6597021018893648], ["Contrast", 0.5218466423129691, 0.562985661685268]], [["Rotate", 0.4401753067886466, 0.055198255925702475], ["Rotate", 0.3702153509335602, 0.5821574425474759]], [["TranslateY", 0.6714729117832363, 0.7145542887432927], ["Equalize", 0.0023263758097700205, 0.25837341854887885]], [["Cutout", 0.3159707561240235, 0.19539664199170742], ["TranslateY", 0.8702824829864558, 0.5832348977243467]], [["AutoContrast", 0.24800812729140026, 0.08017301277245716], ["Brightness", 0.5775505849482201, 0.4905904775616114]], [["Color", 0.4143517886294533, 0.8445937742921498], ["ShearY", 0.28688910858536587, 0.17539366839474402]], [["Brightness", 0.6341134194059947, 0.43683815933640435], ["Brightness", 0.3362277685899835, 0.4612826163288225]], [["Sharpness", 0.4504035748829761, 0.6698294470467474], ["Posterize", 0.9610055612671645, 0.21070714173174876]], [["Posterize", 0.19490421920029832, 0.7235798208354267], ["Rotate", 0.8675551331308305, 0.46335565746433094]], [["Color", 0.35097958351003306, 0.42199181561523186], ["Invert", 0.914112788087429, 0.44775583211984815]], [["Cutout", 0.223575616055454, 0.6328591417299063], ["TranslateY", 0.09269465212259387, 0.5101073959070608]], [["Rotate", 0.3315734525975911, 0.9983593458299167], ["Sharpness", 0.12245416662856974, 0.6258689139914664]], [["ShearY", 0.696116760180471, 0.6317805202283014], ["Color", 0.847501151593963, 0.4440116609830195]], [["Solarize", 0.24945891607225948, 0.7651150206105561], ["Cutout", 0.7229677092930331, 0.12674657348602494]], [["TranslateX", 0.43461945065713675, 0.06476571036747841], ["Color", 0.6139316940180952, 0.7376264330632316]], [["Invert", 0.1933003530637138, 0.4497819016184308], ["Invert", 0.18391634069983653, 0.3199769100951113]], [["Color", 0.20418296626476137, 0.36785101882029814], ["Posterize", 0.624658293920083, 0.8390081535735991]], [["Sharpness", 0.5864963540530814, 0.586672446690273], ["Posterize", 0.1980280647652339, 0.222114611452575]], [["Invert", 0.3543654961628104, 0.5146369635250309], ["Equalize", 0.40751271919434434, 0.4325310837291978]], [["ShearY", 0.22602859359451877, 0.13137880879778158], ["Posterize", 0.7475029061591305, 0.803900538461099]], [["Sharpness", 0.12426276165599924, 0.5965912716602046], ["Invert", 0.22603903038966913, 0.4346802001255868]], [["TranslateY", 0.010307035630661765, 0.16577665156754046], ["Posterize", 0.4114319141395257, 0.829872913683949]], [["TranslateY", 0.9353069865746215, 0.5327821671247214], ["Color", 0.16990443486261103, 0.38794866007484197]], [["Cutout", 0.1028174322829021, 0.3955952903458266], ["ShearY", 0.4311995281335693, 0.48024695395374734]], [["Posterize", 0.1800334334284686, 0.0548749478418862], ["Brightness", 0.7545808536793187, 0.7699080551646432]], [["Color", 0.48695305373084197, 0.6674269768464615], ["ShearY", 0.4306032279086781, 0.06057690550239343]], [["Brightness", 0.4919399683825053, 0.677338905806407], ["Brightness", 0.24112708387760828, 0.42761103121157656]], [["Posterize", 0.4434818644882532, 0.9489450593207714], ["Posterize", 0.40957675116385955, 0.015664946759584186]], [["Posterize", 0.41307949855153797, 0.6843276552020272], ["Rotate", 0.8003545094091291, 0.7002300783416026]], [["Color", 0.7038570031770905, 0.4697612983649519], ["Sharpness", 0.9700016496081002, 0.25185103545948884]], [["AutoContrast", 0.714641656154856, 0.7962423001719023], ["Sharpness", 0.2410097684093468, 0.5919171048019731]], [["TranslateX", 0.8101567644494714, 0.7156447005337443], ["Solarize", 0.5634727831229329, 0.8875158446846]], [["Sharpness", 0.5335258857303261, 0.364743126378182], ["Color", 0.453280875871377, 0.5621962714743068]], [["Cutout", 0.7423678127672542, 0.7726370777867049], ["Invert", 0.2806161382641934, 0.6021111986900146]], [["TranslateY", 0.15190341320343761, 0.3860373175487939], ["Cutout", 0.9980805818665679, 0.05332384819400854]], [["Posterize", 0.36518675678786605, 0.2935819027397963], ["TranslateX", 0.26586180351840005, 0.303641300745208]], [["Brightness", 0.19994509744377761, 0.90813953707639], ["Equalize", 0.8447217761297836, 0.3449396603478335]], [["Sharpness", 0.9294773669936768, 0.999713346583839], ["Brightness", 0.1359744825665662, 0.1658489221872924]], [["TranslateX", 0.11456529257659381, 0.9063795878367734], ["Equalize", 0.017438134319894553, 0.15776887259743755]], [["ShearX", 0.9833726383270114, 0.5688194948373335], ["Equalize", 0.04975615490994345, 0.8078130016227757]], [["Brightness", 0.2654654830488695, 0.8989789725280538], ["TranslateX", 0.3681535065952329, 0.36433345713161036]], [["Rotate", 0.04956524209892327, 0.5371942433238247], ["ShearY", 0.0005527499145153714, 0.56082571605602]], [["Rotate", 0.7918337108932019, 0.5906896260060501], ["Posterize", 0.8223967034091191, 0.450216998388943]], [["Color", 0.43595106766978337, 0.5253013785221605], ["Sharpness", 0.9169421073531799, 0.8439997639348893]], [["TranslateY", 0.20052300197155504, 0.8202662448307549], ["Sharpness", 0.2875792108435686, 0.6997181624527842]], [["Color", 0.10568089980973616, 0.3349467065132249], ["Brightness", 0.13070947282207768, 0.5757725013960775]], [["AutoContrast", 0.3749999712869779, 0.6665578760607657], ["Brightness", 0.8101178402610292, 0.23271946112218125]], [["Color", 0.6473605933679651, 0.7903409763232029], ["ShearX", 0.588080941572581, 0.27223524148254086]], [["Cutout", 0.46293361616697304, 0.7107761001833921], ["AutoContrast", 0.3063766931658412, 0.8026114219854579]], [["Brightness", 0.7884854981520251, 0.5503669863113797], ["Brightness", 0.5832456158675261, 0.5840349298921661]], [["Solarize", 0.4157539625058916, 0.9161905834309929], ["Sharpness", 0.30628197221802017, 0.5386291658995193]], [["Sharpness", 0.03329610069672856, 0.17066672983670506], ["Invert", 0.9900547302690527, 0.6276238841220477]], [["Solarize", 0.551015648982762, 0.6937104775938737], ["Color", 0.8838491591064375, 0.31596634380795385]], [["AutoContrast", 0.16224182418148447, 0.6068227969351896], ["Sharpness", 0.9599468096118623, 0.4885289719905087]], [["TranslateY", 0.06576432526133724, 0.6899544605400214], ["Posterize", 0.2177096480169678, 0.9949164789616582]], [["Solarize", 0.529820544480292, 0.7576047224165541], ["Sharpness", 0.027047878909321643, 0.45425231553970685]], [["Sharpness", 0.9102526010473146, 0.8311987141993857], ["Invert", 0.5191838751826638, 0.6906136644742229]], [["Solarize", 0.4762773516008588, 0.7703654263842423], ["Color", 0.8048437792602289, 0.4741523094238038]], [["Sharpness", 0.7095055508594206, 0.7047344238075169], ["Sharpness", 0.5059623654132546, 0.6127255499234886]], [["TranslateY", 0.02150725921966186, 0.3515764519224378], ["Posterize", 0.12482170119714735, 0.7829851754051393]], [["Color", 0.7983830079184816, 0.6964694521670339], ["Brightness", 0.3666527856286296, 0.16093151636495978]], [["AutoContrast", 0.6724982375829505, 0.536777706678488], ["Sharpness", 0.43091754837597646, 0.7363240924241439]], [["Brightness", 0.2889770401966227, 0.4556557902380539], ["Sharpness", 0.8805303296690755, 0.6262218017754902]], [["Sharpness", 0.5341939854581068, 0.6697109101429343], ["Rotate", 0.6806606655137529, 0.4896914517968317]], [["Sharpness", 0.5690509737059344, 0.32790632371915096], ["Posterize", 0.7951894258661069, 0.08377850335209162]], [["Color", 0.6124132978216081, 0.5756485920709012], ["Brightness", 0.33053544654445344, 0.23321841707002083]], [["TranslateX", 0.0654795026615917, 0.5227246924310244], ["ShearX", 0.2932320531132063, 0.6732066478183716]], [["Cutout", 0.6226071187083615, 0.01009274433736012], ["ShearX", 0.7176799968189801, 0.3758780240463811]], [["Rotate", 0.18172339508029314, 0.18099184896819184], ["ShearY", 0.7862658331645667, 0.295658135767252]], [["Contrast", 0.4156099177015862, 0.7015784500878446], 
                        ["Sharpness", 0.6454135310009, 0.32335858947955287]], [["Color", 0.6215885089922037, 0.6882673235388836], ["Brightness", 0.3539881732605379, 0.39486736455795496]], [["Invert", 0.8164816716866418, 0.7238192000817796], ["Sharpness", 0.3876355847343607, 0.9870077619731956]], [["Brightness", 0.1875628712629315, 0.5068115936257], ["Sharpness", 0.8732419122060423, 0.5028019258530066]], [["Sharpness", 0.6140734993408259, 0.6458239834366959], ["Rotate", 0.5250107862824867, 0.533419456933602]], [["Sharpness", 0.5710893143725344, 0.15551651073007305], ["ShearY", 0.6548487860151722, 0.021365083044319146]], [["Color", 0.7610250354649954, 0.9084452893074055], ["Brightness", 0.6934611792619156, 0.4108071412071374]], [["ShearY", 0.07512550098923898, 0.32923768385754293], ["ShearY", 0.2559588911696498, 0.7082337365398496]], [["Cutout", 0.5401319018926146, 0.004750568603408445], ["ShearX", 0.7473354415031975, 0.34472481968368773]], [["Rotate", 0.02284154583679092, 0.1353450082435801], ["ShearY", 0.8192458031684238, 0.2811653613473772]], [["Contrast", 0.21142896718139154, 0.7230739568811746], ["Sharpness", 0.6902690582665707, 0.13488436112901683]], [["Posterize", 0.21701219600958138, 0.5900695769640687], ["Rotate", 0.7541095031505971, 0.5341162375286219]], [["Posterize", 0.5772853064792737, 0.45808311743269936], ["Brightness", 0.14366050177823675, 0.4644871239446629]], [["Cutout", 0.8951718842805059, 0.4970074074310499], ["Equalize", 0.3863835903119882, 0.9986531042150006]], [["Equalize", 0.039411354473938925, 0.7475477254908457], ["Sharpness", 0.8741966378291861, 0.7304822679596362]], [["Solarize", 0.4908704265218634, 0.5160677350249471], ["Color", 0.24961813832742435, 0.09362352627360726]], [["Rotate", 7.870457075154214e-05, 0.8086950025500952], ["Solarize", 0.10200484521793163, 0.12312889222989265]], [["Contrast", 0.8052564975559727, 0.3403813036543645], ["Solarize", 0.7690158533600184, 0.8234626822018851]], [["AutoContrast", 0.680362728854513, 0.9415320040873628], ["TranslateY", 0.5305871824686941, 0.8030609611614028]], [["Cutout", 0.1748050257378294, 0.06565343731910589], ["TranslateX", 0.1812738872339903, 0.6254461448344308]], [["Brightness", 0.4230502644722749, 0.3346463682905031], ["ShearX", 0.19107198973659312, 0.6715789128604919]], [["ShearX", 0.1706528684548394, 0.7816570201200446], ["TranslateX", 0.494545185948171, 0.4710810058360291]], [["TranslateX", 0.42356251508933324, 0.23865307292867322], ["TranslateX", 0.24407503619326745, 0.6013778508137331]], [["AutoContrast", 0.7719512185744232, 0.3107905373009763], ["ShearY", 0.49448082925617176, 0.5777951230577671]], [["Cutout", 0.13026983827940525, 0.30120438757485657], ["Brightness", 0.8857896834516185, 0.7731541459513939]], [["AutoContrast", 0.6422800349197934, 0.38637401090264556], ["TranslateX", 0.25085431400995084, 0.3170642592664873]], [["Sharpness", 0.22336654455367122, 0.4137774852324138], ["ShearY", 0.22446851054920894, 0.518341735882535]], [["Color", 0.2597579403253848, 0.7289643913060193], ["Sharpness", 0.5227416670468619, 0.9239943674030637]], [["Cutout", 0.6835337711563527, 0.24777620448593812], ["AutoContrast", 0.37260245353051846, 0.4840361183247263]], [["Posterize", 0.32756602788628375, 0.21185124493743707], ["ShearX", 0.25431504951763967, 0.19585996561416225]], [["AutoContrast", 0.07930627591849979, 0.5719381348340309], ["AutoContrast", 0.335512380071304, 0.4208050118308541]], [["Rotate", 0.2924360268257798, 0.5317629242879337], ["Sharpness", 0.4531050021499891, 0.4102650087199528]], [["Equalize", 0.5908862210984079, 0.468742362277498], ["Brightness", 0.08571766548550425, 0.5629320703375056]], [["Cutout", 0.52751122383816, 0.7287774744737556], ["Equalize", 0.28721628275296274, 0.8075179887475786]], [["AutoContrast", 0.24208377391366226, 0.34616549409607644], ["TranslateX", 0.17454707403766834, 0.5278055700078459]], [["Brightness", 0.5511881924749478, 0.999638675514418], ["Equalize", 0.14076197797220913, 0.2573030693317552]], [["ShearX", 0.668731433926434, 0.7564253049646743], ["Color", 0.63235486543845, 0.43954436063340785]], [["ShearX", 0.40511960873276237, 0.5710419512142979], ["Contrast", 0.9256769948746423, 0.7461350716211649]], [["Cutout", 0.9995917204023061, 0.22908419326246265], ["TranslateX", 0.5440902956629469, 0.9965570051216295]], [["Color", 0.22552987172228894, 0.4514558960849747], ["Sharpness", 0.638058150559443, 0.9987829481002615]], [["Contrast", 0.5362775837534763, 0.7052133185951871], ["ShearY", 0.220369845547023, 0.7593922994775721]], [["ShearX", 0.0317785822935219, 0.775536785253455], ["TranslateX", 0.7939510227015061, 0.5355620618496535]], [["Cutout", 0.46027969917602196, 0.31561199122527517], ["Color", 0.06154066467629451, 0.5384660000729091]], [["Sharpness", 0.7205483743301113, 0.552222392539886], ["Posterize", 0.5146496404711752, 0.9224333144307473]], [["ShearX", 0.00014547730356910538, 0.3553954298642108], ["TranslateY", 0.9625736029090676, 0.57403418640424]], [["Posterize", 0.9199917903297341, 0.6690259107633706], ["Posterize", 0.0932558110217602, 0.22279303372106138]], [["Invert", 0.25401453476874863, 0.3354329544078385], ["Posterize", 0.1832673201325652, 0.4304718799821412]], [["TranslateY", 0.02084122674367607, 0.12826181437197323], ["ShearY", 0.655862534043703, 0.3838330909470975]], [["Contrast", 0.35231797644104523, 0.3379356652070079], ["Cutout", 0.19685599014304822, 0.1254328595280942]], [["Sharpness", 0.18795594984191433, 0.09488678946484895], ["ShearX", 0.33332876790679306, 0.633523782574133]], [["Cutout", 0.28267175940290246, 0.7901991550267817], ["Contrast", 0.021200195312951198, 0.4733128702798515]], [["ShearX", 0.966231043411256, 0.7700673327786812], ["TranslateX", 0.7102390777763321, 0.12161245817120675]], [["Cutout", 0.5183324259533826, 0.30766086003013055], ["Color", 0.48399078150128927, 0.4967477809069189]], [["Sharpness", 0.8160855187385873, 0.47937658961644], ["Posterize", 0.46360395447862535, 0.7685454058155061]], [["ShearX", 0.10173571421694395, 0.3987290690178754], ["TranslateY", 0.8939980277379345, 0.5669994143735713]], [["Posterize", 0.6768089584801844, 0.7113149244621721], ["Posterize", 0.054896856043358935, 0.3660837250743921]], [["AutoContrast", 0.5915576211896306, 0.33607718177676493], ["Contrast", 0.3809408206617828, 0.5712201773913784]], [["AutoContrast", 0.012321347472748323, 0.06379072432796573], ["Rotate", 0.0017964439160045656, 0.7598026295973337]], [["Contrast", 0.6007100085192627, 0.36171972473370206], ["Invert", 0.09553573684975913, 0.12218510774295901]], [["AutoContrast", 0.32848604643836266, 0.2619457656206414], ["Invert", 0.27082113532501784, 0.9967965642293485]], [["AutoContrast", 0.6156282120903395, 0.9422706516080884], ["Sharpness", 0.4215509247379262, 0.4063347716503587]], [["Solarize", 0.25059210436331264, 0.7215305521159305], ["Invert", 0.1654465185253614, 0.9605851884186778]], [["AutoContrast", 0.4464438610980994, 0.685334175815482], ["Cutout", 0.24358625461158645, 0.4699066834058694]], [["Rotate", 0.5931657741857909, 0.6813978655574067], ["AutoContrast", 0.9259100547738681, 0.4903201223870492]], [["Color", 0.8203976071280751, 0.9777824466585101], ["Posterize", 0.4620669369254169, 0.2738895968716055]], [["Contrast", 0.13754352055786848, 0.3369433962088463], ["Posterize", 0.48371187792441916, 0.025718004361451302]], [["Rotate", 0.5208233630704999, 0.1760188899913535], ["TranslateX", 0.49753461392937226, 0.4142935276250922]], [["Cutout", 0.5967418240931212, 0.8028675552639539], ["Cutout", 0.20021854152659121, 0.19426330549590076]], [["ShearY", 0.549583567386676, 0.6601326640171705], ["Cutout", 0.6111813470383047, 0.4141935587984994]], [["Brightness", 0.6354891977535064, 0.31591459747846745], ["AutoContrast", 0.7853952208711621, 0.6555861906702081]], [["AutoContrast", 0.7333725370546154, 0.9919410576081586], ["Cutout", 0.9984177877923588, 0.2938253683694291]], [["Color", 0.33219296307742263, 0.6378995578424113], ["AutoContrast", 0.15432820754183288, 0.7897899838932103]], [["Contrast", 0.5905289460222578, 0.8158577207653422], ["Cutout", 0.3980284381203051, 0.43030531250317217]], [["TranslateX", 0.452093693346745, 0.5251475931559115], ["Rotate", 0.991422504871258, 0.4556503729269001]], [["Color", 0.04560406292983776, 0.061574671308480766], ["Brightness", 0.05161079440128734, 0.6718398142425688]], [["Contrast", 0.02913302416506853, 0.14402056093217708], ["Rotate", 0.7306930378774588, 0.47088249057922094]], [["Solarize", 0.3283072384190169, 0.82680847744367], ["Invert", 0.21632614168418854, 0.8792241691482687]], [["Equalize", 0.4860808352478527, 0.9440534949023064], ["Cutout", 0.31395897639184694, 0.41805859306017523]], [["Rotate", 0.2816043232522335, 0.5451282807926706], ["Color", 0.7388520447173302, 0.7706503658143311]], [["Color", 0.9342776719536201, 0.9039981381514299], ["Rotate", 0.6646389177840164, 0.5147917008383647]], [["Cutout", 0.08929430082050335, 0.22416445996932374], ["Posterize", 0.454485751267457, 0.500958345348237]], [["TranslateX", 0.14674201106374488, 0.7018633472428202], ["Sharpness", 0.6128796723832848, 0.743535235614809]], [["TranslateX", 0.5189900164469432, 0.6491132403587601], ["Contrast", 0.26309555778227806, 0.5976857969656114]], [["Solarize", 0.23569808291972655, 0.3315781686591778], ["ShearY", 0.07292078937544964, 0.7460326987587573]], [["ShearY", 0.7090542757477153, 0.5246437008439621], ["Sharpness", 0.9666919148538443, 0.4841687888767071]], [["Solarize", 0.3486952615189488, 0.7012877201721799], ["Invert", 0.1933387967311534, 0.9535472742828175]], [["AutoContrast", 0.5393460721514914, 0.6924005011697713], ["Cutout", 0.16988156769247176, 0.3667207571712882]], [["Rotate", 0.5815329514554719, 0.5390406879316949], ["AutoContrast", 0.7370538341589625, 0.7708822194197815]], [["Color", 0.8463701017918459, 0.9893491045831084], ["Invert", 0.06537367901579016, 0.5238468509941635]], [["Contrast", 0.8099771812443645, 0.39371603893945184], ["Posterize", 0.38273629875646487, 0.46493786058573966]], 
                        [["Color", 0.11164686537114032, 0.6771450570033168], ["Posterize", 0.27921361289661406, 0.7214300893597819]], [["Contrast", 0.5958265906571906, 0.5963959447666958], ["Sharpness", 0.2640889223630885, 0.3365870842641453]], [["Color", 0.255634146724125, 0.5610029792926452], ["ShearY", 0.7476893976084721, 0.36613194760395557]], [["ShearX", 0.2167581882130063, 0.022978065071245002], ["TranslateX", 0.1686864409720319, 0.4919575435512007]], [["Solarize", 0.10702753776284957, 0.3954707963684698], ["Contrast", 0.7256100635368403, 0.48845259655719686]], [["Sharpness", 0.6165615058519549, 0.2624079463213861], ["ShearX", 0.3804820351860919, 0.4738994677544202]], [["TranslateX", 0.18066394808448177, 0.8174509422318228], ["Solarize", 0.07964569396290502, 0.45495935736800974]], [["Sharpness", 0.2741884021129658, 0.9311045302358317], ["Cutout", 0.0009101326429323388, 0.5932102256756948]], [["Rotate", 0.8501796375826188, 0.5092564038282137], ["Brightness", 0.6520146983999912, 0.724091283316938]], [["Brightness", 0.10079744898900078, 0.7644088017429471], ["AutoContrast", 0.33540215138213575, 0.1487538541758792]], [["ShearY", 0.10632545944757177, 0.9565164562996977], ["Rotate", 0.275833816849538, 0.6200731548023757]], [["Color", 0.6749819274397422, 0.41042188598168844], ["AutoContrast", 0.22396590966461932, 0.5048018491863738]], [["Equalize", 0.5044277111650255, 0.2649182381110667], ["Brightness", 0.35715133289571355, 0.8653260893016869]], [["Cutout", 0.49083594426355326, 0.5602781291093129], ["Posterize", 0.721795488514384, 0.5525847430754974]], [["Sharpness", 0.5081835448947317, 0.7453323423804428], ["TranslateX", 0.11511932212234266, 0.4337766796030984]], [["Solarize", 0.3817050641766593, 0.6879004573473403], ["Invert", 0.0015041436267447528, 0.9793134066888262]], [["AutoContrast", 0.5107410439697935, 0.8276720355454423], ["Cutout", 0.2786270701864015, 0.43993387208414564]], [["Rotate", 0.6711202569428987, 0.6342930903972932], ["Posterize", 0.802820231163559, 0.42770002619222053]], [["Color", 0.9426854321337312, 0.9055431782458764], ["AutoContrast", 0.3556422423506799, 0.2773922428787449]], [["Contrast", 0.10318991257659992, 0.30841372533347416], ["Posterize", 0.4202264962677853, 0.05060395018085634]], [["Invert", 0.549305630337048, 0.886056156681853], ["Cutout", 0.9314157033373055, 0.3485836940307909]], [["ShearX", 0.5642891775895684, 0.16427372934801418], ["Invert", 0.228741164726475, 0.5066345406806475]], [["ShearY", 0.5813123201003086, 0.33474363490586106], ["Equalize", 0.11803439432255824, 0.8583936440614798]], [["Sharpness", 0.1642809706111211, 0.6958675237301609], ["ShearY", 0.5989560762277414, 0.6194018060415276]], [["Rotate", 0.05092104774529638, 0.9358045394527796], ["Cutout", 0.6443254331615441, 0.28548414658857657]], [["Brightness", 0.6986036769232594, 0.9618046340942727], ["Sharpness", 0.5564490243465492, 0.6295231286085622]], [["Brightness", 0.42725649792574105, 0.17628028916784244], ["Equalize", 0.4425109360966546, 0.6392872650036018]], [["ShearY", 0.5758622795525444, 0.8773349286588288], ["ShearX", 0.038525646435423666, 0.8755366512394268]], [["Sharpness", 0.3704459924265827, 0.9236361456197351], ["Color", 0.6379842432311235, 0.4548767717224531]], [["Contrast", 0.1619523824549347, 0.4506528800882731], ["AutoContrast", 0.34513874426188385, 0.3580290330996726]], [["Contrast", 0.728699731513527, 0.6932238009822878], ["Brightness", 0.8602917375630352, 0.5341445123280423]], [["Equalize", 0.3574552353044203, 0.16814745124536548], ["Rotate", 0.24191717169379262, 0.3279497108179034]], [["ShearY", 0.8567478695576244, 0.37746117240238164], ["ShearX", 0.9654125389830487, 0.9283047610798827]], [["ShearY", 0.4339052480582405, 0.5394548246617406], ["Cutout", 0.5070570647967001, 0.7846286976687882]], [["AutoContrast", 0.021620100406875065, 0.44425839772845227], ["AutoContrast", 0.33978157614075183, 0.47716564815092244]], [["Contrast", 0.9727600659025666, 0.6651758819229426], ["Brightness", 0.9893133904996626, 0.39176397622636105]], [["Equalize", 0.283428620586305, 0.18727922861893637], ["Rotate", 0.3556063466797136, 0.3722839913107821]], [["ShearY", 0.7276172841941864, 0.4834188516302227], ["ShearX", 0.010783217950465884, 0.9756458772142235]], [["ShearY", 0.2901753295101581, 0.5684700238749064], ["Cutout", 0.655585564610337, 0.9490071307790201]], [["AutoContrast", 0.008507193981450278, 0.4881150103902877], ["AutoContrast", 0.6561989723231185, 0.3715071329838596]], [["Contrast", 0.7702505530948414, 0.6961371266519999], ["Brightness", 0.9953051630261895, 0.3861962467326121]], [["Equalize", 0.2805270012472756, 0.17715406116880994], ["Rotate", 0.3111256593947474, 0.15824352183820073]], [["Brightness", 0.9888680802094193, 0.4856236485253163], ["ShearX", 0.022370252047332284, 0.9284975906226682]], [["ShearY", 0.4065719044318099, 0.7468528006921563], ["AutoContrast", 0.19494427109708126, 0.8613186475174786]], [["AutoContrast", 0.023296727279367765, 0.9170949567425306], ["AutoContrast", 0.11663051100921168, 0.7908646792175343]], [["AutoContrast", 0.7335191671571732, 0.4958357308292425], ["Color", 0.7964964008349845, 0.4977687544324929]], [["ShearX", 0.19905221600021472, 0.3033081933150046], ["Equalize", 0.9383410219319321, 0.3224669877230161]], [["ShearX", 0.8265450331466404, 0.6509091423603757], ["Sharpness", 0.7134181178748723, 0.6472835976443643]], [["ShearY", 0.46962439525486044, 0.223433110541722], ["Rotate", 0.7749806946212373, 0.5337060376916906]], [["Posterize", 0.1652499695106796, 0.04860659068586126], ["Brightness", 0.6644577712782511, 0.4144528269429337]], [["TranslateY", 0.6220449565731829, 0.4917495676722932], ["Posterize", 0.6255000355409635, 0.8374266890984867]], [["AutoContrast", 0.4887160797052227, 0.7106426020530529], ["Sharpness", 0.7684218571497236, 0.43678474722954763]], [["Invert", 0.13178101535845366, 0.8301141976359813], ["Color", 0.002820877424219378, 0.49444413062487075]], [["TranslateX", 0.9920683666478188, 0.5862245842588877], ["Posterize", 0.5536357075855376, 0.5454300367281468]], [["Brightness", 0.8150181219663427, 0.1411060258870707], ["Sharpness", 0.8548823004164599, 0.77008691072314]], [["Brightness", 0.9580478020413399, 0.7198667636628974], ["ShearY", 0.8431585033377366, 0.38750016565010803]], [["Solarize", 0.2331505347152334, 0.25754361489084787], ["TranslateY", 0.447431373734262, 0.5782399531772253]], [["TranslateY", 0.8904927998691309, 0.25872872455072315], ["AutoContrast", 0.7129888139716263, 0.7161603231650524]], [["ShearY", 0.6336216800247362, 0.5247508616674911], ["Cutout", 0.9167315119726633, 0.2060557387978919]], [["ShearX", 0.001661782345968199, 0.3682225725445044], ["Solarize", 0.12303352043754572, 0.5014989548584458]], [["Brightness", 0.9723625105116246, 0.6555444729681099], ["Contrast", 0.5539208721135375, 0.7819973409318487]], [["Equalize", 0.3262607499912611, 0.0006745572802121513], ["Contrast", 0.35341551623767103, 0.36814689398886347]], [["ShearY", 0.7478539900243613, 0.37322078030129185], ["TranslateX", 0.41558847793529247, 0.7394615158544118]], [["Invert", 0.13735541232529067, 0.5536403864332143], ["Cutout", 0.5109718190377135, 0.0447509485253679]], [["AutoContrast", 0.09403602327274725, 0.5909250807862687], ["ShearY", 0.53234060616395, 0.5316981359469398]], [["ShearX", 0.5651922367876323, 0.6794110241313183], ["Posterize", 0.7431624856363638, 0.7896861463783287]], [["Brightness", 0.30949179379286806, 0.7650569096019195], ["Sharpness", 0.5461629122105034, 0.6814369444005866]], [["Sharpness", 0.28459340191768434, 0.7802208350806028], ["Rotate", 0.15097973114238117, 0.5259683294104645]], [["ShearX", 0.6430803693700531, 0.9333735880102375], ["Contrast", 0.7522209520030653, 0.18831747966185058]], [["Contrast", 0.4219455937915647, 0.29949769435499646], ["Color", 0.6925322933509542, 0.8095523885795443]], [["ShearX", 0.23553236193043048, 0.17966207900468323], ["AutoContrast", 0.9039700567886262, 0.21983629944639108]], [["ShearX", 0.19256223146671514, 0.31200739880443584], ["Sharpness", 0.31962196883294713, 0.6828107668550425]], [["Cutout", 0.5947690279080912, 0.21728220253899178], ["Rotate", 0.6757188879871141, 0.489460599679474]], [["ShearY", 0.18365897125470526, 0.3988571115918058], ["Brightness", 0.7727489489504, 0.4790369956329955]], [["Contrast", 0.7090301084131432, 0.5178303607560537], ["ShearX", 0.16749258277688506, 0.33061773301592356]], [["ShearX", 0.3706690885419934, 0.38510677124319415], ["AutoContrast", 0.8288356276501032, 0.16556487668770264]], [["TranslateY", 0.16758043046445614, 0.30127092823893986], ["Brightness", 0.5194636577132354, 0.6225165310621702]], [["Cutout", 0.6087289363049726, 0.10439287037803044], ["Rotate", 0.7503452083033819, 0.7425316019981433]], [["ShearY", 0.24347189588329932, 0.5554979486672325], ["Brightness", 0.9468115239174161, 0.6132449358023568]], [["Brightness", 0.7144508395807994, 0.4610594769966929], ["ShearX", 0.16466683833092968, 0.3382903812375781]], [["Sharpness", 0.27743648684265465, 0.17200038071656915], ["Color", 0.47404262107546236, 0.7868991675614725]], [["Sharpness", 0.8603993513633618, 0.324604728411791], ["TranslateX", 0.3331597130403763, 0.9369586812977804]], [["Color", 0.1535813630595832, 0.4700116846558207], ["Color", 0.5435647971896318, 0.7639291483525243]], [["Brightness", 0.21486188101947656, 0.039347277341450576], ["Cutout", 0.7069526940684954, 0.39273934115015696]], [["ShearY", 0.7267130888840517, 0.6310800726389485], ["AutoContrast", 0.662163190824139, 0.31948540372237766]], [["ShearX", 0.5123132117185981, 0.1981015909438834], ["AutoContrast", 0.9009347363863067, 0.26790399126924036]], [["Brightness", 0.24245061453231648, 0.2673478678291436], ["ShearX", 0.31707976089283946, 0.6800582845544948]], [["Cutout", 0.9257780138367764, 0.03972673526848819], ["Rotate", 0.6807858944518548, 0.46974332280612097]], [["ShearY", 0.1543443071262312, 0.6051682587030671], ["Brightness", 0.9758203119828304, 0.4941406868162414]], [["Contrast", 0.07578049236491124, 0.38953819133407647], 
                        ["ShearX", 0.20194918288164293, 0.4141510791947318]], [["Color", 0.27826402243792286, 0.43517491081531157], ["AutoContrast", 0.6159269026143263, 0.2021846783488046]], [["AutoContrast", 0.5039377966534692, 0.19241507605941105], ["Invert", 0.5563931144385394, 0.7069728937319112]], [["Sharpness", 0.19031632433810566, 0.26310171056096743], ["Color", 0.4724537593175573, 0.6715201448387876]], [["ShearY", 0.2280910467786642, 0.33340559088059313], ["ShearY", 0.8858560034869303, 0.2598627441471076]], [["ShearY", 0.07291814128021593, 0.5819462692986321], ["Cutout", 0.27605696060512147, 0.9693427371868695]], [["Posterize", 0.4249871586563321, 0.8256952014328607], ["Posterize", 0.005907466926447169, 0.8081353382152597]], [["Brightness", 0.9071305290601128, 0.4781196213717954], ["Posterize", 0.8996214311439275, 0.5540717376630279]], [["Brightness", 0.06560728936236392, 0.9920627849065685], ["TranslateX", 0.04530789794044952, 0.5318568944702607]], [["TranslateX", 0.6800263601084814, 0.4611536772507228], ["Rotate", 0.7245888375283157, 0.0914772551375381]], [["Sharpness", 0.879556061897963, 0.42272481462067535], ["TranslateX", 0.4600350422524085, 0.5742175429334919]], [["AutoContrast", 0.5005776243176145, 0.22597121331684505], ["Invert", 0.10763286370369299, 0.6841782704962373]], [["Sharpness", 0.7422908472000116, 0.6850324203882405], ["TranslateX", 0.3832914614128403, 0.34798646673324896]], [["ShearY", 0.31939465302679326, 0.8792088167639516], ["Brightness", 0.4093604352811235, 0.21055483197261338]], [["AutoContrast", 0.7447595860998638, 0.19280222555998586], ["TranslateY", 0.317754779431227, 0.9983454520593591]], [["Equalize", 0.27706973689750847, 0.6447455020660622], ["Contrast", 0.5626579126863761, 0.7920049962776781]], [["Rotate", 0.13064369451773816, 0.1495367590684905], ["Sharpness", 0.24893941981801215, 0.6295943894521504]], [["ShearX", 0.6856269993063254, 0.5167938584189854], ["Sharpness", 0.24835352574609537, 0.9990550493102627]], [["AutoContrast", 0.461654115871693, 0.43097388896245004], ["Cutout", 0.366359682416437, 0.08011826474215511]], [["AutoContrast", 0.993892672935951, 0.2403608711236933], ["ShearX", 0.6620817870694181, 0.1744814077869482]], [["ShearY", 0.6396747719986443, 0.15031017143644265], ["Brightness", 0.9451954879495629, 0.26490678840264714]], [["Color", 0.19311480787397262, 0.15712300697448575], ["Posterize", 0.05391448762015258, 0.6943963643155474]], [["Sharpness", 0.6199669674684085, 0.5412492335319072], ["Invert", 0.14086213450149815, 0.2611850277919339]], [["Posterize", 0.5533129268803405, 0.5332478159319912], ["ShearX", 0.48956244029096635, 0.09223930853562916]], [["ShearY", 0.05871590849449765, 0.19549715278943228], ["TranslateY", 0.7208521362741379, 0.36414003004659434]], [["ShearY", 0.7316263417917531, 0.0629747985768501], ["Contrast", 0.036359793501448245, 0.48658745414898386]], [["Rotate", 0.3301497610942963, 0.5686622043085637], ["ShearX", 0.40581487555676843, 0.5866127743850192]], [["ShearX", 0.6679039628249283, 0.5292270693200821], ["Sharpness", 0.25901391739310703, 0.9778360586541461]], [["AutoContrast", 0.27373222012596854, 0.14456771405730712], ["Contrast", 0.3877220783523938, 0.7965158941894336]], [["Solarize", 0.29440905483979096, 0.06071633809388455], ["Equalize", 0.5246736285116214, 0.37575084834661976]], [["TranslateY", 0.2191269464520395, 0.7444942293988484], ["Posterize", 0.3840878524812771, 0.31812671711741247]], [["Solarize", 0.25159267140731356, 0.5833264622559661], ["Brightness", 0.07552262572348738, 0.33210648549288435]], [["AutoContrast", 0.9770099298399954, 0.46421915310428197], ["AutoContrast", 0.04707358934642503, 0.24922048012183493]], [["Cutout", 0.5379685806621965, 0.02038212605928355], ["Brightness", 0.5900728303717965, 0.28807872931416956]], [["Sharpness", 0.11596624872886108, 0.6086947716949325], ["AutoContrast", 0.34876470059667525, 0.22707897759730578]], [["Contrast", 0.276545513135698, 0.8822580384226156], ["Rotate", 0.04874027684061846, 0.6722214281612163]], [["ShearY", 0.595839851757025, 0.4389866852785822], ["Equalize", 0.5225492356128832, 0.2735290854063459]], [["Sharpness", 0.9918029636732927, 0.9919926583216121], ["Sharpness", 0.03672376137997366, 0.5563865980047012]], [["AutoContrast", 0.34169589759999847, 0.16419911552645738], ["Invert", 0.32995953043129234, 0.15073174739720568]], [["Posterize", 0.04600255098477292, 0.2632612790075844], ["TranslateY", 0.7852153329831825, 0.6990722310191976]], [["AutoContrast", 0.4414653815356372, 0.2657468780017082], ["Posterize", 0.30647061536763337, 0.3688222724948656]], [["Contrast", 0.4239361091421837, 0.6076562806342001], ["Cutout", 0.5780707784165284, 0.05361325256745192]], [["Sharpness", 0.7657895907855394, 0.9842407321667671], ["Sharpness", 0.5416352696151596, 0.6773681575200902]], [["AutoContrast", 0.13967381098331305, 0.10787258006315015], ["Posterize", 0.5019536507897069, 0.9881978222469807]], [["Brightness", 0.030528346448984903, 0.31562058762552847], ["TranslateY", 0.0843808140595676, 0.21019213305350526]], [["AutoContrast", 0.6934579165006736, 0.2530484168209199], ["Rotate", 0.0005751408130693636, 0.43790043943210005]], [["TranslateX", 0.611258547664328, 0.25465240215894935], ["Sharpness", 0.5001446909868196, 0.36102204109889413]], [["Contrast", 0.8995127327150193, 0.5493190695343996], ["Brightness", 0.242708780669213, 0.5461116653329015]], [["AutoContrast", 0.3751825351022747, 0.16845985803896962], ["Cutout", 0.25201103287363663, 0.0005893331783358435]], [["ShearX", 0.1518985779435941, 0.14768180777304504], ["Color", 0.85133530274324, 0.4006641163378305]], [["TranslateX", 0.5489668255504668, 0.4694591826554948], ["Rotate", 0.1917354490155893, 0.39993269385802177]], [["ShearY", 0.6689267479532809, 0.34304285013663577], ["Equalize", 0.24133154048883143, 0.279324043138247]], [["Contrast", 0.3412544002099494, 0.20217358823930232], ["Color", 0.8606984790510235, 0.14305503544676373]], [["Cutout", 0.21656155695311988, 0.5240101349572595], ["Brightness", 0.14109877717636352, 0.2016827341210295]], [["Sharpness", 0.24764371218833872, 0.19655480259925423], ["Posterize", 0.19460398862039913, 0.4975414350200679]], [["Brightness", 0.6071850094982323, 0.7270716448607151], ["Solarize", 0.111786402398499, 0.6325641684614275]], [["Contrast", 0.44772949532200856, 0.44267502710695955], ["AutoContrast", 0.360117506402693, 0.2623958228760273]], [["Sharpness", 0.8888131688583053, 0.936897400764746], ["Sharpness", 0.16080674198274894, 0.5681119841445879]], [["AutoContrast", 0.8004456226590612, 0.1788600469525269], ["Brightness", 0.24832285390647374, 0.02755350284841604]], [["ShearY", 0.06910320102646594, 0.26076407321544054], ["Contrast", 0.8633703022354964, 0.38968514704043056]], [["AutoContrast", 0.42306251382780613, 0.6883260271268138], ["Rotate", 0.3938724346852023, 0.16740881249086037]], [["Contrast", 0.2725343884286728, 0.6468194318074759], ["Sharpness", 0.32238942646494745, 0.6721149242783824]], [["AutoContrast", 0.942093919956842, 0.14675331481712853], ["Posterize", 0.5406276708262192, 0.683901182218153]], [["Cutout", 0.5386811894643584, 0.04498833938429728], ["Posterize", 0.17007257321724775, 0.45761177118620633]], [["Contrast", 0.13599408935104654, 0.53282738083886], ["Solarize", 0.26941667995081114, 0.20958261079465895]], [["Color", 0.6600788518606634, 0.9522228302165842], ["Invert", 0.0542722262516899, 0.5152431169321683]], [["Contrast", 0.5328934819727553, 0.2376220512388278], ["Posterize", 0.04890422575781711, 0.3182233123739474]], [["AutoContrast", 0.9289628064340965, 0.2976678437448435], ["Color", 0.20936893798507963, 0.9649612821434217]], [["Cutout", 0.9019423698575457, 0.24002036989728096], ["Brightness", 0.48734445615892974, 0.047660899809176316]], [["Sharpness", 0.09347824275711591, 0.01358686275590612], ["Posterize", 0.9248539660538934, 0.4064232632650468]], [["Brightness", 0.46575675383704634, 0.6280194775484345], ["Invert", 0.17276207634499413, 0.21263495428839635]], [["Brightness", 0.7238014711679732, 0.6178946027258592], ["Equalize", 0.3815496086340364, 0.07301281068847276]], [["Contrast", 0.754557393588416, 0.895332753570098], ["Color", 0.32709957750707447, 0.8425486003491515]], [["Rotate", 0.43406698081696576, 0.28628263254953723], ["TranslateY", 0.43949548709125374, 0.15927082198238685]], [["Brightness", 0.0015838339831640708, 0.09341692553352654], ["AutoContrast", 0.9113966907329718, 0.8345900469751112]], [["ShearY", 0.46698796308585017, 0.6150701348176804], ["Invert", 0.14894062704815722, 0.2778388046184728]], [["Color", 0.30360499169455957, 0.995713092016834], ["Contrast", 0.2597016288524961, 0.8654420870658932]], [["Brightness", 0.9661642031891435, 0.7322006407169436], ["TranslateY", 0.4393502786333408, 0.33934762664274265]], [["Color", 0.9323638351992302, 0.912776309755293], ["Brightness", 0.1618274755371618, 0.23485741708056307]], [["Color", 0.2216470771158821, 0.3359240197334976], ["Sharpness", 0.6328691811471494, 0.6298393874452548]], [["Solarize", 0.4772769142265505, 0.7073470698713035], ["ShearY", 0.2656114148206966, 0.31343097010487253]], [["Solarize", 0.3839017339304234, 0.5985505779429036], ["Equalize", 0.002412059429196589, 0.06637506181196245]], [["Contrast", 0.12751196553017863, 0.46980311434237976], ["Sharpness", 0.3467487455865491, 0.4054907610444406]], [["AutoContrast", 0.9321813669127206, 0.31328471589533274], ["Rotate", 0.05801738717432747, 0.36035756254444273]], [["TranslateX", 0.52092390458353, 0.5261722561643886], ["Contrast", 0.17836804476171306, 0.39354333443158535]], [["Posterize", 0.5458100909925713, 0.49447244994482603], ["Brightness", 0.7372536822363605, 0.5303409097463796]], [["Solarize", 0.1913974941725724, 0.5582966653986761], ["Equalize", 0.020733669175727026, 0.9377467166472878]], [["Equalize", 0.16265732137763889, 0.5206282340874929], ["Sharpness", 0.2421533133595281, 0.506389065871883]], [["AutoContrast", 0.9787324801448523, 0.24815051941486466], ["Rotate", 0.2423487151245957, 0.6456493129745148]], 
                        [["TranslateX", 0.6809867726670327, 0.6949687002397612], ["Contrast", 0.16125673359747458, 0.7582679978218987]], [["Posterize", 0.8212000950994955, 0.5225012157831872], ["Brightness", 0.8824891856626245, 0.4499216779709508]], [["Solarize", 0.12061313332505218, 0.5319371283368052], ["Equalize", 0.04120865969945108, 0.8179402157299602]], [["Rotate", 0.11278256686005855, 0.4022686554165438], ["ShearX", 0.2983451019112792, 0.42782525461812604]], [["ShearY", 0.8847385513289983, 0.5429227024179573], ["Rotate", 0.21316428726607445, 0.6712120087528564]], [["TranslateX", 0.46448081241068717, 0.4746090648963252], ["Brightness", 0.19973580961271142, 0.49252862676553605]], [["Posterize", 0.49664100539481526, 0.4460713166484651], ["Brightness", 0.6629559985581529, 0.35192346529003693]], [["Color", 0.22710733249173676, 0.37943185764616194], ["ShearX", 0.015809774971472595, 0.8472080190835669]], [["Contrast", 0.4187366322381491, 0.21621979869256666], ["AutoContrast", 0.7631045030367304, 0.44965231251615134]], [["Sharpness", 0.47240637876720515, 0.8080091811749525], ["Cutout", 0.2853425420104144, 0.6669811510150936]], [["Posterize", 0.7830320527127324, 0.2727062685529881], ["Solarize", 0.527834000867504, 0.20098218845222998]], [["Contrast", 0.366380535288225, 0.39766001659663075], ["Cutout", 0.8708808878088891, 0.20669525734273086]], [["ShearX", 0.6815427281122932, 0.6146858582671569], ["AutoContrast", 0.28330622372053493, 0.931352024154997]], [["AutoContrast", 0.8668174463154519, 0.39961453880632863], ["AutoContrast", 0.5718557712359253, 0.6337062930797239]], [["ShearY", 0.8923152519411871, 0.02480062504737446], ["Cutout", 0.14954159341231515, 0.1422219808492364]], [["Rotate", 0.3733718175355636, 0.3861928572224287], ["Sharpness", 0.5651126520194574, 0.6091103847442831]], [["Posterize", 0.8891714191922857, 0.29600154265251016], ["TranslateY", 0.7865351723963945, 0.5664998548985523]], [["TranslateX", 0.9298214806998273, 0.729856565052017], ["AutoContrast", 0.26349082482341846, 0.9638882609038888]], [["Sharpness", 0.8387378377527128, 0.42146721129032494], ["AutoContrast", 0.9860522000876452, 0.4200699464169384]], [["ShearY", 0.019609159303115145, 0.37197835936879514], ["Cutout", 0.22199340461754258, 0.015932573201085848]], [["Rotate", 0.43871085583928443, 0.3283504258860078], ["Sharpness", 0.6077702068037776, 0.6830305349618742]], [["Contrast", 0.6160211756538094, 0.32029451083389626], ["Cutout", 0.8037631428427006, 0.4025688837399259]], [["TranslateY", 0.051637820936985435, 0.6908417834391846], ["Sharpness", 0.7602756948473368, 0.4927111506643095]], [["Rotate", 0.4973618638052235, 0.45931479729281227], ["TranslateY", 0.04701789716427618, 0.9408779705948676]], [["Rotate", 0.5214194592768602, 0.8371249272013652], ["Solarize", 0.17734812472813338, 0.045020798970228315]], [["ShearX", 0.7457999920079351, 0.19025612553075893], ["Sharpness", 0.5994846101703786, 0.5665094068864229]], [["Contrast", 0.6172655452900769, 0.7811432139704904], ["Cutout", 0.09915620454670282, 0.3963692287596121]], [["TranslateX", 0.2650112299235817, 0.7377261946165307], ["AutoContrast", 0.5019539734059677, 0.26905046992024506]], [["Contrast", 0.6646299821370135, 0.41667784809592945], ["Cutout", 0.9698457154992128, 0.15429001887703997]], [["Sharpness", 0.9467079029475773, 0.44906457469098204], ["Cutout", 0.30036908747917396, 0.4766149689663106]], [["Equalize", 0.6667517691051055, 0.5014839828447363], ["Solarize", 0.4127890336820831, 0.9578274770236529]], [["Cutout", 0.6447384874120834, 0.2868806107728985], ["Cutout", 0.4800990488106021, 0.4757538246206956]], [["Solarize", 0.12560195032363236, 0.5557473475801568], ["Equalize", 0.019957161871490228, 0.5556797187823773]], [["Contrast", 0.12607637375759484, 0.4300633627435161], ["Sharpness", 0.3437273670109087, 0.40493203127714417]], [["AutoContrast", 0.884353334807183, 0.5880138314357569], ["Rotate", 0.9846032404597116, 0.3591877296622974]], [["TranslateX", 0.6862295865975581, 0.5307482119690076], ["Contrast", 0.19439251187251982, 0.3999195825722808]], [["Posterize", 0.4187641835025246, 0.5008988942651585], ["Brightness", 0.6665805605402482, 0.3853288204214253]], [["Posterize", 0.4507470690013903, 0.4232437206624681], ["TranslateX", 0.6054107416317659, 0.38123828040922203]], [["AutoContrast", 0.29562338573283276, 0.35608605102687474], ["TranslateX", 0.909954785390274, 0.20098894888066549]], [["Contrast", 0.6015278411777212, 0.6049140992035096], ["Cutout", 0.47178713636517855, 0.5333747244651914]], [["TranslateX", 0.490851976691112, 0.3829593925141144], ["Sharpness", 0.2716675173824095, 0.5131696240367152]], [["Posterize", 0.4190558294646337, 0.39316689077269873], ["Rotate", 0.5018526072725914, 0.295712490156129]], [["AutoContrast", 0.29624715560691617, 0.10937329832409388], ["Posterize", 0.8770505275992637, 0.43117765012206943]], [["Rotate", 0.6649970092751698, 0.47767131373391974], ["ShearX", 0.6257923540490786, 0.6643337040198358]], [["Sharpness", 0.5553620705849509, 0.8467799429696928], ["Cutout", 0.9006185811918932, 0.3537270716262]], [["ShearY", 0.0007619678283789788, 0.9494591850536303], ["Invert", 0.24267733654007673, 0.7851608409575828]], [["Contrast", 0.9730916198112872, 0.404670123321921], ["Sharpness", 0.5923587793251186, 0.7405792404430281]], [["Cutout", 0.07393909593373034, 0.44569630026328344], ["TranslateX", 0.2460593252211425, 0.4817527814541055]], [["Brightness", 0.31058654119340867, 0.7043749950260936], ["ShearX", 0.7632161538947713, 0.8043681264908555]], [["AutoContrast", 0.4352334371415373, 0.6377550087204297], ["Rotate", 0.2892714673415678, 0.49521052050510556]], [["Equalize", 0.509071051375276, 0.7352913414974414], ["ShearX", 0.5099959429711828, 0.7071566714593619]], [["Posterize", 0.9540506532512889, 0.8498853304461906], ["ShearY", 0.28199061357155397, 0.3161715627214629]], [["Posterize", 0.6740855359097433, 0.684004694936616], ["Posterize", 0.6816720350737863, 0.9654766942980918]], [["Solarize", 0.7149344531717328, 0.42212789795181643], ["Brightness", 0.686601460864528, 0.4263050070610551]], [["Cutout", 0.49577164991501, 0.08394890892056037], ["Rotate", 0.5810369852730606, 0.3320732965776973]], [["TranslateY", 0.1793755480490623, 0.6006520265468684], ["Brightness", 0.3769016576438939, 0.7190746300828186]], [["TranslateX", 0.7226363597757153, 0.3847027238123509], ["Brightness", 0.7641713191794035, 0.36234003077512544]], [["TranslateY", 0.1211227055347106, 0.6693523474608023], ["Brightness", 0.13011180247738063, 0.5126647617294864]], [["Equalize", 0.1501070550869129, 0.0038548909451806557], ["Posterize", 0.8266535939653881, 0.5502199643499207]], [["Sharpness", 0.550624117428359, 0.2023044586648523], ["Brightness", 0.06291556314780017, 0.7832635398703937]], [["Color", 0.3701578205508141, 0.9051537973590863], ["Contrast", 0.5763972727739397, 0.4905511239739898]], [["Rotate", 0.7678527224046323, 0.6723066265307555], ["Solarize", 0.31458533097383207, 0.38329324335154524]], [["Brightness", 0.292050127929522, 0.7047582807953063], ["ShearX", 0.040541891910333805, 0.06639328601282746]], [["TranslateY", 0.4293891393238555, 0.6608516902234284], ["Sharpness", 0.7794685477624004, 0.5168044063408147]], [["Color", 0.3682450402286552, 0.17274523597220048], ["ShearY", 0.3936056470397763, 0.5702597289866161]], [["Equalize", 0.43436990310624657, 0.9207072627823626], ["Contrast", 0.7608688260846083, 0.4759023148841439]], [["Brightness", 0.7926088966143935, 0.8270093925674497], ["ShearY", 0.4924174064969461, 0.47424347505831244]], [["Contrast", 0.043917555279430476, 0.15861903591675125], ["ShearX", 0.30439480405505853, 0.1682659341098064]], [["TranslateY", 0.5598255583454538, 0.721352536005039], ["Posterize", 0.9700921973303752, 0.6882015184440126]], [["AutoContrast", 0.3620887415037668, 0.5958176322317132], ["TranslateX", 0.14213781552733287, 0.6230799786459947]], [["Color", 0.490366889723972, 0.9863152892045195], ["Color", 0.817792262022319, 0.6755656429452775]], [["Brightness", 0.7030707021937771, 0.254633187122679], ["Color", 0.13977318232688843, 0.16378180123959793]], [["AutoContrast", 0.2933247831326118, 0.6283663376211102], ["Sharpness", 0.85430478154147, 0.9753613184208796]], [["Rotate", 0.6674299955457268, 0.48571208708018976], ["Contrast", 0.47491370175907016, 0.6401079552479657]], [["Sharpness", 0.37589579644127863, 0.8475131989077025], ["TranslateY", 0.9985149867598191, 0.057815729375099975]], [["Equalize", 0.0017194373841596389, 0.7888361311461602], ["Contrast", 0.6779293670669408, 0.796851411454113]], [["TranslateY", 0.3296782119072306, 0.39765117357271834], ["Sharpness", 0.5890554357001884, 0.6318339473765834]], [["Posterize", 0.25423810893163856, 0.5400430289894207], ["Sharpness", 0.9273643918988342, 0.6480913470982622]], [["Cutout", 0.850219975768305, 0.4169812455601289], ["Solarize", 0.5418755745870089, 0.5679666650495466]], [["Brightness", 0.008881361977310959, 0.9282562314720516], ["TranslateY", 0.7736066471553994, 0.20041167606029642]], [["Brightness", 0.05382537581401925, 0.6405265501035952], ["Contrast", 0.30484329473639593, 0.5449338155734242]], [["Color", 0.613257119787967, 0.4541503912724138], ["Brightness", 0.9061572524724674, 0.4030159294447347]], [["Brightness", 0.02739111568942537, 0.006028056532326534], ["ShearX", 0.17276751958646486, 0.05967365780621859]], [["TranslateY", 0.4376298213047888, 0.7691816164456199], ["Sharpness", 0.8162292718857824, 0.6054926462265117]], [["Color", 0.37963069679121214, 0.5946919433483344], ["Posterize", 0.08485417284005387, 0.5663580913231766]], [["Equalize", 0.49785780226818316, 0.9999137109183761], ["Sharpness", 0.7685879484682496, 0.6260846154212211]], [["AutoContrast", 0.4190931409670763, 0.2374852525139795], ["Posterize", 0.8797422264608563, 0.3184738541692057]], [["Rotate", 0.7307269024632872, 0.41523609600701106], ["ShearX", 0.6166685870692289, 0.647133807748274]], [["Sharpness", 0.5633713231039904, 0.8276694754755876], ["Cutout", 0.8329340776895764, 0.42656043027424073]], 
                        [["ShearY", 0.14934828370884312, 0.8622510773680372], ["Invert", 0.25925989086863277, 0.8813283584888576]], [["Contrast", 0.9457071292265932, 0.43228655518614034], ["Sharpness", 0.8485316947644338, 0.7590298998732413]], [["AutoContrast", 0.8386103589399184, 0.5859583131318076], ["Solarize", 0.466758711343543, 0.9956215363818983]], [["Rotate", 0.9387133710926467, 0.19180564509396503], ["Rotate", 0.5558247609706255, 0.04321698692007105]], [["ShearX", 0.3608716600695567, 0.15206159451532864], ["TranslateX", 0.47295292905710146, 0.5290760596129888]], [["TranslateX", 0.8357685981547495, 0.5991305115727084], ["Posterize", 0.5362929404188211, 0.34398525441943373]], [["ShearY", 0.6751984031632811, 0.6066293622133011], ["Contrast", 0.4122723990263818, 0.4062467515095566]], [["Color", 0.7515349936021702, 0.5122124665429213], ["Contrast", 0.03190514292904123, 0.22903520154660545]], [["Contrast", 0.5448962625054385, 0.38655673938910545], ["AutoContrast", 0.4867400684894492, 0.3433111101096984]], [["Rotate", 0.0008372434310827959, 0.28599951781141714], ["Equalize", 0.37113686925530087, 0.5243929348114981]], [["Color", 0.720054993488857, 0.2010177651701808], ["TranslateX", 0.23036196506059398, 0.11152764304368781]], [["Cutout", 0.859134208332423, 0.6727345740185254], ["ShearY", 0.02159833505865088, 0.46390076266538544]], [["Sharpness", 0.3428232157391428, 0.4067874527486514], ["Brightness", 0.5409415136577347, 0.3698432231874003]], [["Solarize", 0.27303978936454776, 0.9832186173589548], ["ShearY", 0.08831127213044043, 0.4681870331149774]], [["TranslateY", 0.2909309268736869, 0.4059460811623174], ["Sharpness", 0.6425125139803729, 0.20275737203293587]], [["Contrast", 0.32167626214661627, 0.28636162794046977], ["Invert", 0.4712405253509603, 0.7934644799163176]], [["Color", 0.867993060896951, 0.96574321666213], ["Color", 0.02233897320328512, 0.44478933557303063]], [["AutoContrast", 0.1841254751814967, 0.2779992148017741], ["Color", 0.3586283093530607, 0.3696246850445087]], [["Posterize", 0.2052935984046965, 0.16796913860308244], ["ShearX", 0.4807226832843722, 0.11296747254563266]], [["Cutout", 0.2016411266364791, 0.2765295444084803], ["Brightness", 0.3054112810424313, 0.695924264931216]], [["Rotate", 0.8405872184910479, 0.5434142541450815], ["Cutout", 0.4493615138203356, 0.893453735250007]], [["Contrast", 0.8433310507685494, 0.4915423577963278], ["ShearX", 0.22567799557913246, 0.20129892537008834]], [["Contrast", 0.045954277103674224, 0.5043900167190442], ["Cutout", 0.5552992473054611, 0.14436447810888237]], [["AutoContrast", 0.7719296115130478, 0.4440417544621306], ["Sharpness", 0.13992809206158283, 0.7988278670709781]], [["Color", 0.7838574233513952, 0.5971351401625151], ["TranslateY", 0.13562290583925385, 0.2253039635819158]], [["Cutout", 0.24870301109385806, 0.6937886690381568], ["TranslateY", 0.4033400068952813, 0.06253378991880915]], [["TranslateX", 0.0036059390486775644, 0.5234723884081843], ["Solarize", 0.42724862530733526, 0.8697702564187633]], [["Equalize", 0.5446026737834311, 0.9367992979112202], ["ShearY", 0.5943478903735789, 0.42345889214100046]], [["ShearX", 0.18611885697957506, 0.7320849092947314], ["ShearX", 0.3796416430900566, 0.03817761920009881]], [["Posterize", 0.37636778506979124, 0.26807924785236537], ["Brightness", 0.4317372554383255, 0.5473346211870932]], [["Brightness", 0.8100436240916665, 0.3817612088285007], ["Brightness", 0.4193974619003253, 0.9685902764026623]], [["Contrast", 0.701776402197012, 0.6612786008858009], ["Color", 0.19882787177960912, 0.17275597188875483]], [["Color", 0.9538303302832989, 0.48362384535228686], ["ShearY", 0.2179980837345602, 0.37027290936457313]], [["TranslateY", 0.6068028691503798, 0.3919346523454841], ["Cutout", 0.8228303342563138, 0.18372280287814613]], [["Equalize", 0.016416758802906828, 0.642838949194916], ["Cutout", 0.5761717838655257, 0.7600661153497648]], [["Color", 0.9417761826818639, 0.9916074035986558], ["Equalize", 0.2524209308597042, 0.6373703468715077]], [["Brightness", 0.75512589439513, 0.6155072321007569], ["Contrast", 0.32413476940254515, 0.4194739830159837]], [["Sharpness", 0.3339450765586968, 0.9973297539194967], ["AutoContrast", 0.6523930242124429, 0.1053482471037186]], [["ShearX", 0.2961391955838801, 0.9870036064904368], ["ShearY", 0.18705025965909403, 0.4550895821154484]], [["TranslateY", 0.36956447983807883, 0.36371471767143543], ["Sharpness", 0.6860051967688487, 0.2850190720087796]], [["Cutout", 0.13017742151902967, 0.47316674150067195], ["Invert", 0.28923829959551883, 0.9295585654924601]], [["Contrast", 0.7302368472279086, 0.7178974949876642], ["TranslateY", 0.12589674152030433, 0.7485392909494947]], [["Color", 0.6474693117772619, 0.5518269515590674], ["Contrast", 0.24643004970708016, 0.3435581358079418]], [["Contrast", 0.5650327855750835, 0.4843031798040887], ["Brightness", 0.3526684005761239, 0.3005305004600969]], [["Rotate", 0.09822284968122225, 0.13172798244520356], ["Equalize", 0.38135066977857157, 0.5135129123554154]], [["Contrast", 0.5902590645585712, 0.2196062383730596], ["ShearY", 0.14188379126120954, 0.1582612142182743]], [["Cutout", 0.8529913814417812, 0.89734031211874], ["Color", 0.07293767043078672, 0.32577659205278897]], [["Equalize", 0.21401668971453247, 0.040015259500028266], ["ShearY", 0.5126400895338797, 0.4726484828276388]], [["Brightness", 0.8269430025954498, 0.9678362841865166], ["ShearY", 0.17142069814830432, 0.4726727848289514]], [["Brightness", 0.699707089334018, 0.2795501395789335], ["ShearX", 0.5308818178242845, 0.10581814221896294]], [["Equalize", 0.32519644258946145, 0.15763390340309183], ["TranslateX", 0.6149090364414208, 0.7454832565718259]], [["AutoContrast", 0.5404508567155423, 0.7472387762067986], ["Equalize", 0.05649876539221024, 0.5628180219887216]]]
            
            policies = []
            for policy in self.policies:
                if not "Cutout" == policy[0][0] and not "Cutout" == policy[1][0]:
                    policies.append(policy)
            self.policies = policies

            # # test
            # self.policies = [[
            #                 ["ShearX", 0.22404644446773492, 0.6508620171913467], 
            #                 ["ShearY", 0.14143816458479197, 0.513124791615952], 
            #                 ["TranslateX", 0.47295292905710146, 0.5290760596129888], 
            #                 ["TranslateY", 0.49865058747734736, 0.4352676987103321], 
            #                 ["Rotate", 0.550289356406774, 0.38419022293237126], 
            #                 ["AutoContrast", 0.5748186910788027, 0.8185482599354216], 
            #                 ["Invert", 0.6872393349333636, 0.9307694335024579], 
            #                 ["Equalize", 0.3863835903119882, 0.9986531042150006], 
            #                 ["Solarize", 0.12560195032363236, 0.5557473475801568],
            #                 ["Posterize", 0.5188375788270497, 0.9863648925446865],
            #                 ["Contrast", 0.47219838080793364, 0.8228524484275648],
            #                 ["Color", 0.11164686537114032, 0.6771450570033168],
            #                 ["Brightness", 0.3539881732605379, 0.39486736455795496],
            #                 ["Sharpness", 0.9290316227291179, 0.9788406212603302],
            #                 ["Cutout", 0.2853425420104144, 0.6669811510150936]
            #                 ]]

    def get_transform(self, img):

        return AutoAugTransform(self.policies)
# Bacon

def apply_transform_gens(transform_gens, img):
    """
    Apply a list of :class:`TransformGen` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the image, because a subsequent transform may
    need the output of the previous one.

    Args:
        transform_gens (list): list of :class:`TransformGen` instance to
            be applied.
        img (ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        ndarray: the transformed image
        TransformList: contain the transforms that's used.
    """
    for g in transform_gens:
        assert isinstance(g, TransformGen), g

    check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img)
        assert isinstance(
            tfm, Transform
        ), "TransformGen {} must return an instance of Transform! Got {} instead".format(g, tfm)
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)
