import glob
import numpy as np
import os
from PIL import Image
import json

def load_LIP_semantic(image_dir, gt_dir, bg_dir=None):

    cache_dir = os.path.join(image_dir, "../cache.json")

    if not os.path.isfile(cache_dir):
        ret = []
        for image_file in glob.glob(os.path.join(image_dir, "*.jpg")):
            
            if "INVALID" in gt_dir:
                label_file = None
            else:
                label_file = gt_dir + image_file.replace(".jpg", ".png").replace(image_dir, "")
                assert os.path.isfile(label_file)

            image = np.asarray(Image.open(image_file))
            
            ret.append(
                {
                    "file_name": image_file,
                    "sem_seg_file_name": label_file,
                    "bg_file_dir": bg_dir,
                    "height": image.shape[0],
                    "width": image.shape[1],
                }
            )
    else:
        with open(cache_dir, "r") as f:
            cache = json.load(f)
            ret = []
            for r in cache.values():
                ret.append(r)

    return ret