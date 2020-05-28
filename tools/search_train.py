# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments, run
from ray.tune.experiment import convert_to_experiment_list
import logging
from collections import OrderedDict
import torch
from hyperopt import hp
from collections import defaultdict
import time
import numpy as np
import datetime
import json
import os
import cv2

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import MattingEvaluator, inference_on_dataset
from detectron2.utils.autoaug import augment_list, Accumulator
from detectron2.utils.logger import log_every_n_seconds
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


class PreTrainer(DefaultTrainer):

    def __init__(self, cfg, k_th=0):
        
        super().__init__(cfg)

        # count for train iteration to get sub dataset
        self.train_iter = 0

        # basic configure
        self.cfg                = cfg
        self.k_th               = k_th
        self.K_fold             = cfg.DATASETS.KFOLD
        

    # the Step1 train main flow
    def train(self):

        logger = logging.getLogger("detectron2.trainer")
        logger.info("{}_th fold {} sub datasets".format(self.k_th, self.K_fold))

        # pretrain with initial augmentation policies
        logger.info("Step1: pretrain model")
        super().train()


    # for Step1 train
    def run_step(self):

        self.model.train()
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # use sub dataset, for fast auto augmentation
        # only change the data that read, did not chanage the iteratation count
        while not self.train_iter % self.K_fold == self.k_th:
            next(self._data_loader_iter)
            self.train_iter += 1

        # get data
        data = next(self._data_loader_iter)
        self.train_iter += 1
        data_time = time.perf_counter() - start

        # forward
        loss_dict, res = self.model(data)
        losses = loss_dict["loss_sem_seg"]
        
        try:
            self._detect_anomaly(losses, loss_dict)
        except:
            self.write_image(res, data)

            assert 0, "Loss became infinite or NaN at iteration={}".format(self.train_iter)

        # log
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        # back propogation
        self.optimizer.zero_grad()
        losses.backward()

        # optimize
        self.optimizer.step()

        if not self.train_iter % 1000:
            self.write_image(res, data)


    @classmethod
    # for Step1 evaluation
    def test(cls, cfg, model, evaluators=None):
        
        logger = logging.getLogger("detectron2.trainer")
        
        # inference in all datasets
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:

            # logger.info("{}_th fold, dataset {} inference".format(cls.k_th, dataset_name))

            # get dataloader
            data_loader = build_detection_test_loader(cfg, dataset_name)
            # get evaluator
            evaluator = MattingEvaluator(dataset_name)

            # inference
            results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)
            

        if len(results) == 1:
            results = list(results.values())[0]

        return results

    def write_image(self, res, data):
        
        cv2.imwrite("/home/pgding/project/LIP/detectron2_bdd/tools/output/train_out/{}_".format(self.train_iter)+res[0].split("/")[-1].replace(".jpg", "_train_og.jpg"), data[0]["image"].numpy().transpose(1,2,0))
        cv2.imwrite("/home/pgding/project/LIP/detectron2_bdd/tools/output/train_out/{}_".format(self.train_iter)+res[0].split("/")[-1].replace(".jpg", "_train_gt.jpg"), res[1].astype(np.uint8))
        cv2.imwrite("/home/pgding/project/LIP/detectron2_bdd/tools/output/train_out/{}_".format(self.train_iter)+res[0].split("/")[-1].replace(".jpg", "_train_pr.jpg"), res[2])


def search_debug(augment, reporter):
    import random
    reporter(minus_loss=random.random(), top1_valid=random.random(), done=True)  


def search_func(model, K_fold, augment, reporter):

    logger = logging.getLogger("detectron2.trainer")

    # get constant configuration
    cfg        = augment["cfg"]
    k_th       = augment["k_th"]
    K_fold     = augment["K_fold"]
    num_op     = augment["num_op"]
    ops_list   = augment["ops_list"]
    num_policy = augment["num_policy"]

    logger.info("start search {}_th of {} sub datasets".format(k_th, K_fold)) 

    # get allocated augmentation policies
    policies = policy_decoder(augment, num_policy, num_op, ops_list)

    # inference without backward
    results = SearchTrainer.test_policies(cfg, model, policies, k_th, K_fold)

    # losses and accuracies in all datasets
    losses, mious = [], []
    for result in results.values():
        losses.append(result["loss"])
        mious.append(result["miou"])
    
    metrics = Accumulator()
    metrics.add_dict({
        'minus_loss': -1 * np.mean(losses),
        'correct': np.mean(mious),
    })
    
    del losses, mious, policies, results

    # report for optimization
    reporter(minus_loss=metrics['minus_loss'], top1_valid=metrics['correct'], done=True)


class SearchTrainer:

    def __init__(self, cfg, resume_f, k_th=0):

        self.cfg                 = cfg
        self.smoke_test          = cfg.AUTOAUG.SMOKE_TEST
        self.num_search          = cfg.AUTOAUG.NUM_SEARCH
        self.num_policy          = cfg.AUTOAUG.NUM_POLICY
        self.num_op              = cfg.AUTOAUG.NUM_OP
        self.num_final_policies  = cfg.AUTOAUG.NUM_FINAL_POLICIES
        self.K_fold              = cfg.DATASETS.KFOLD
        self.metric              = cfg.AUTOAUG.METRIC
        self.mode                = cfg.AUTOAUG.MODE
        # resource                 = cfg.AUTOAUG.RESOURCES_PER_TRIAL.split(":")
        self.resources_per_trial = {"gpu":0.5, "cpu":4}#{resource[0]: int(resource[1])}

        self.k_th               = k_th

        self.ops_list           = augment_list()
        self.space              = self._get_search_space()
        self.model              = self._load_model(resume_f)


    # make sure the model evaluate right and check the HyperOptionSearch can run
    def network_debug(self):
 
        logger = logging.getLogger("detectron2.trainer")

        # inference
        SearchTrainer.test_policies(self.cfg, self.model, None, self.k_th, self.K_fold)

        # search by explore and exploit
        logger.info("Step2: search best policies")
        name = "search_fold%d" % (self.k_th)
        register_trainable(name, lambda augs, rpt: search_debug(augs, rpt))

        # search algorithm
        algo = HyperOptSearch(self.space, max_concurrent=4*20, metric=self.metric, mode=self.mode)  # top1_valid or minus_loss

        # experiments configuration
        exp_config = {
            name: {
                'run': name,
                'num_samples': 4,
                'resources_per_trial': self.resources_per_trial,
                'stop': {'training_iteration': self.num_policy},
                'config': {"cfg": self.cfg, "k_th": self.k_th, "K_fold": self.K_fold}
            }
        }

        # bayes optimization search
        # results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, raise_on_failed_trial=False)
        results = run(  
            convert_to_experiment_list(exp_config),
            name=name, 
            search_alg=algo,
            resources_per_trial=self.resources_per_trial,
            return_trials=True,
            verbose=0,
            queue_trials=True,
            raise_on_failed_trial=False,
        )
        
        # sort
        results = [x for x in results if x.last_result is not None]
        results = sorted(results, key=lambda x: x.last_result[self.metric], reverse=True)

        return []


    # search by explore and exploit
    def search(self):

        logger = logging.getLogger("detectron2.trainer")
        logger.info("Step2: search best policies")

        name = "search_fold%d" % (self.k_th)
        
        # regist function
        register_trainable(name, lambda augs, rpt: search_func(self.model, self.K_fold, augs, rpt))

        # search algorithm
        algo = HyperOptSearch(self.space, max_concurrent=4*20, metric=self.metric, mode=self.mode)  # max top1_valid or min minus_loss

        # configuration
        exp_config = {
            name: {
                'run': name,
                'num_samples': 40 if self.smoke_test else self.num_search,
                "resources_per_trial": self.resources_per_trial,
                'stop': {'training_iteration': self.num_policy},
                'config': {"cfg":self.cfg, "k_th":self.k_th, "K_fold":self.K_fold, "num_policy":self.num_policy, "num_op":self.num_op, "ops_list":self.ops_list}
            }
        }
        
        # bayes optimization search
        # results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, raise_on_failed_trial=False)
        results = run(  
            convert_to_experiment_list(exp_config),
            name=name, 
            search_alg=algo,
            resources_per_trial=self.resources_per_trial,
            return_trials=True,
            verbose=0,
            queue_trials=True,
            raise_on_failed_trial=False,
        )
        
        # sort
        results = [x for x in results if x.last_result is not None]
        results = sorted(results, key=lambda x: x.last_result[self.metric], reverse=True)

        # get top N policies
        final_policy_set = []
        for result in results[:self.num_final_policies]:
        # for result in results[:self.num_final_policies *5//self.K_fold]:
            # transform result to policies
            final_policy = policy_decoder(result.config, self.num_policy, self.num_op, self.ops_list)
            logger.info('k_th:%d | loss=%.12f top1_valid=%.4f %s' % (self.k_th, result.last_result['minus_loss'], result.last_result['top1_valid'], final_policy))

            final_policy = self._remove_deplicates(final_policy)
            final_policy_set.extend(final_policy)

        return final_policy_set


    # for Step2
    @staticmethod
    def test_policies(cfg, model, policies, k_th, K_fold):
        
        logger = logging.getLogger("detectron2.trainer")

        # inference in all datasets
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:

            logger.info("{}_th fold, dataset {} inference".format(k_th, dataset_name))

            # get dataloader
            mapper = DatasetMapper(cfg, True, policies)
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper)
            # get evaluator
            evaluator = MattingEvaluator(dataset_name)

            # inference
            results[dataset_name] = SearchTrainer.inference(model, data_loader, evaluator, k_th, K_fold)

        return results


    # for test
    @staticmethod
    def inference(model, data_loader, evaluator, k_th, K_fold):

        total = len(data_loader)  # inference data loader must have a fixed length

        logger = logging.getLogger("detectron2.trainer")
        logger.info("Start inference on {} images".format(total))

        num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # 1.initialize evaluator counter
        evaluator.reset()

        num_warmup = min(5*K_fold, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        torch.no_grad()
        model.eval()
        for idx, inputs in enumerate(data_loader):

            # warm up
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            # select sub dataset
            if not idx % K_fold == k_th:
                continue

            start_compute_time = time.perf_counter()

            # 2.evaluate
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            # 3.update evaluator counter
            evaluator.process(inputs, outputs)

            # log
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "{}_th sub_datasets | Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        k_th, idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )
        
        # 4.final evaluate
        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}

        return results


    def _load_model(self, f):
        
        # build model
        model = build_model(self.cfg)
        
        # resume
        DetectionCheckpointer(model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(f, resume=False)
        
        return model


    def _get_search_space(self):

        space = {}
        for i in range(self.num_policy):
            for j in range(self.num_op):
                space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(self.ops_list))))
                space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_%d' % (i, j), 0.0, 1.0)
                space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_%d' % (i, j), 0.0, 1.0)
        
        return space


    def _remove_deplicates(self, policies):

        s = set()
        new_policies = []
        for ops in policies:
            key = []
            for op in ops:
                key.append(op[0])
            key = '_'.join(key)
            if key in s:
                continue
            else:
                s.add(key)
                new_policies.append(ops)

        return new_policies



def policy_decoder(augment, num_policy, num_op, ops_list):

    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = augment['policy_%d_%d' % (i, j)]
            op_prob = augment['prob_%d_%d' % (i, j)]
            op_level = augment['level_%d_%d' % (i, j)]
            
            try:
                ops.append((ops_list[op_idx][0].__name__, op_prob, op_level))
            except:
                assert 0, op_idx
        policies.append(ops)

    return policies


def pretrain(cfg, k, resume):

    # pretrain
    trainer = PreTrainer(cfg, k_th=k)

    trainer.resume_or_load(resume=resume)

    trainer.train()



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


    # devide dataset into [KFOLD]
    debug            = 0
    skip             = args.skip_pretrain
    resume           = args.resume
    pretrain_f       = "./output/autoaug_pre_search/model_final.pth"

    final_policy_set = []
    # TODO: resume logic didnt adapt to the situation of KFOLD>1
    for k in range(cfg.DATASETS.KFOLD):

        resume_f = pretrain_f.replace(".pth", "_{}.pth".format(k))
        policy_f = resume_f.replace("model_final", "final_policies").replace("pth", "json")

        # Step 1 :Pretrain on dataset D_M
        try:
            if skip:
                assert os.path.isfile(resume_f)
                logger.info("skip pretrain {}".format(k))
            else:
                assert 0
        except:
    
            os.system("rm /home/pgding/project/LIP/detectron2_bdd/tools/output/train_out/*")
            os.system("rm /home/pgding/project/LIP/detectron2_bdd/tools/output/infer_out/*")
            
            skip = False
            launch(
                pretrain,
                args.num_gpus,
                num_machines=args.num_machines,
                machine_rank=args.machine_rank,
                dist_url=args.dist_url,
                args=(cfg, k, resume),
            )
            os.system("mv {} {}".format(pretrain_f, resume_f))

        # Step 2: Search policies on dataset D_A
        if skip and os.path.isfile(policy_f):
            logger.info("skip search {}".format(k))
        else:
            search = SearchTrainer(cfg, resume_f, k)
            if not debug:
                policies = search.search()
            else: # debug
                policies = search.network_debug()

            # save result
            final_policy_set.extend(policies)
            with open(policy_f, 'w') as f:
                json.dump(policies, f)

    # save final policies
    with open('./output/autoaug_pre_search/final_policies.json', 'w') as f:
        json.dump(final_policy_set, f)