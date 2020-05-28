import numpy as np
import torch
from collections import OrderedDict

from detectron2.data import MetadataCatalog
from .evaluator import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize


class MattingEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name):

        self._metadata     = MetadataCatalog.get(dataset_name)
        self._cpu_device   = torch.device("cpu")
        self._distributed  = True
        self._ignore_label = 255
        self._N            = 3


    def reset(self):

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self.ave_loss = AverageMeter()


    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):

            pred_segm = output["sem_seg"].to(self._cpu_device)
            pred = torch.max(pred_segm, dim=0)[1].data

            gt = output["gt_semseg"].to(self._cpu_device)
            gt[gt==self._ignore_label] = self._N-1

            assert len(np.unique(gt))<4

            # import cv2
            # gt_tmp = gt.numpy().copy()
            # gt_tmp[gt_tmp==1] = 100
            # gt_tmp[gt_tmp==2] = 255
            # cv2.imwrite("/home/pgding/project/LIP/detectron2_bdd/tools/output/infer_out/"+input["file_name"].split("/")[-1].replace(".jpg", "_infer_gt.jpg"), gt_tmp)
            # cv2.imwrite("/home/pgding/project/LIP/detectron2_bdd/tools/output/infer_out/"+input["file_name"].split("/")[-1].replace(".jpg", "_infer_pr.jpg"), pred.numpy()*255)

            self._conf_matrix += np.bincount(
                self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
            ).reshape(self._N, self._N)

            # for fast auto augment
            loss = output["loss"]
            loss = loss.mean()
            self.ave_loss.update(loss.data.item())


    def evaluate(self):

        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
        
        # for fast auto augmentation
        loss = self.ave_loss.average()

        # calc iou and acc
        acc = np.zeros(self._N-1, dtype=np.float)
        iou = np.zeros(self._N-1, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        # f = open("output/autoaug/miou", "a")
        # import json
        # f.write(json.dumps({"loss": loss, "miou":miou*100, "fiou":fiou*100, "macc":macc*100, "pacc":pacc*100}))
        # f.write("\n")
        return OrderedDict({"loss": loss, "miou":miou*100, "fiou":fiou*100, "macc":macc*100, "pacc":pacc*100})


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg