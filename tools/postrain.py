import logging
from collections import OrderedDict
import json
import cv2
import time
import torch
import datetime
import os
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import MattingEvaluator, inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


class PostTrainer(DefaultTrainer):

    def __init__(self, cfg):
        
        self.policies = self.get_policies()

        super().__init__(cfg)


    def build_train_loader(self, cfg):
        
        mapper = DatasetMapper(cfg, True, self.policies)
        
        return build_detection_train_loader(cfg, mapper) 


    def run_step(self):

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict, _ = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()

        self.optimizer.step()


    def get_policies(self):
        
        # load the final policies
        with open('./output/autoaug_pre_search/final_policies.json', 'r') as f:
            final_policy_set = json.load(f)

        return final_policy_set


def postrain(cfg, resume):

    # postrain
    trainer = PostTrainer(cfg)

    trainer.resume_or_load(resume=resume)

    trainer.train()


def inference(cfg, out_dir):

    # build model
    model = build_model(cfg)
    # resume
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load("./output/autoaug_post_train/model_final.pth", resume=True)

    # data_loader
    mapper = DatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper)

    total = len(data_loader)  # inference data loader must have a fixed length
    
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    torch.no_grad()
    model.eval()
    for idx, inputs in enumerate(data_loader):

        start_compute_time = time.perf_counter()
        outputs = model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_compute_time += time.perf_counter() - start_compute_time

        # log
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_img = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_img > 5:
            total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
            eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
            log_every_n_seconds(
                logging.INFO,
                "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, total, seconds_per_img, str(eta)
                ),
                n=5,
            )
        
        for input, output in zip(inputs, outputs):

            pred_segm = output["sem_seg"].to("cpu")
            pred = torch.max(pred_segm, dim=0)[1].data
            pred = pred.numpy()[:, :, np.newaxis]
            pred = np.dstack((pred, pred, pred))

            cv2.imwrite(out_dir+input["file_name"].split("/")[-1].replace("jpg", "png"), pred*255)



if __name__ == "__main__":

    logger = logging.getLogger("detectron2.trainer")

    args = default_argument_parser().parse_args()
    logger.info("Command Line Args:{}".format(args))

    # Create configs and perform basic setups.
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    if not args.skip_pretrain:
        # Step 3 : Posttrain with policies from scratch
        launch(
            postrain,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(cfg, args.resume),
        )
    
    else:
        out_dir = "/home/pgding/project/LIP/detectron2_bdd/tools/output/infer_out/"
        os.system("rm {}*".format(out_dir))
        inference(cfg, out_dir)