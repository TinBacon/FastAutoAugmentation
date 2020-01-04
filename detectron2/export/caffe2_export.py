# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import copy
import io
import logging
import struct
from typing import List

import mock
import numpy as np
import onnx
import torch
from caffe2.python.onnx.backend import Caffe2Backend as c2
from detectron2.modeling import meta_arch
from detectron2.structures import ImageList

from .c10 import Caffe2Compatible
from .patcher import ROIHeadsPatcher, patch_generalized_rcnn
from .shared import (
    ScopedWS,
    alias,
    check_set_pb_arg,
    construct_init_net_from_params,
    fuse_alias_placeholder,
    get_params_from_init_net,
    group_norm_replace_aten_with_caffe2,
    mock_torch_nn_functional_interpolate,
    remove_reshape_for_fc,
    save_graph,
)


logger = logging.getLogger(__name__)


def _cast_to_f32(f64):
    return struct.unpack("f", struct.pack("f", f64))[0]


def is_valid_model_output_blob(blob):
    return isinstance(blob, np.ndarray)


def set_caffe2_compatible_tensor_mode(model, enable=True):
    def _fn(m):
        if isinstance(m, Caffe2Compatible):
            m.tensor_mode = enable

    model.apply(_fn)


def convert_batched_inputs_to_c2_format(batched_inputs, size_divisibility, device):
    """
    batched_inputs is a list of dicts, each dict has fileds like image,
    height, width, image_id, etc ...
    # In D2, image is as 3D (C, H, W) tensor, all fields are not batched

    This function turn D2 format input to a tuple of Tensors
    """

    assert all(isinstance(x, dict) for x in batched_inputs)
    assert all(x["image"].dim() == 3 for x in batched_inputs)

    images = [x["image"] for x in batched_inputs]
    images = ImageList.from_tensors(images, size_divisibility)

    im_info = []
    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
        target_height = input_per_image.get("height", image_size[0])
        target_width = input_per_image.get("width", image_size[1])  # noqa
        # FIXME: when using im_info, the scale only appies to the height
        # dimension, will this affect accuracy?
        scale = target_height / image_size[0]
        im_info.append([image_size[0], image_size[1], scale])
    im_info = torch.Tensor(im_info)

    return images.tensor.to(device), im_info.to(device)


def caffe2_preprocess_image(self, inputs):
    """
    Override original preprocess_image, which is called inside the forward.
    Normalize, pad and batch the input images.
    """
    data, im_info = inputs
    data = alias(data, "data")
    im_info = alias(im_info, "im_info")
    normalized_data = self.normalizer(data)
    normalized_data = alias(normalized_data, "normalized_data")

    # Pack (data, im_info) into ImageList which is recognized by self.inference.
    images = ImageList(tensor=normalized_data, image_sizes=im_info)

    return images


@contextlib.contextmanager
def mock_preprocess_image(instance):
    with mock.patch.object(
        type(instance),
        "preprocess_image",
        autospec=True,
        side_effect=caffe2_preprocess_image,
    ) as mocked_func:
        yield
    assert mocked_func.call_count > 0


class Caffe2GeneralizedRCNN(Caffe2Compatible, torch.nn.Module):
    def __init__(self, cfg, torch_model):
        """
        Note: it modifies torch_model in-place.
        """
        super(Caffe2GeneralizedRCNN, self).__init__()
        assert isinstance(torch_model, meta_arch.GeneralizedRCNN)
        self._wrapped_model = patch_generalized_rcnn(torch_model)
        self.eval()
        # self.tensor_mode = False
        set_caffe2_compatible_tensor_mode(self, True)

        self.roi_heads_patcher = ROIHeadsPatcher(cfg, self._wrapped_model.roi_heads)

    def get_tensors_input(self, batched_inputs):
        return convert_batched_inputs_to_c2_format(
            batched_inputs,
            self._wrapped_model.backbone.size_divisibility,
            self._wrapped_model.device,
        )

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"GeneralizedRCNN")
        # NOTE: maybe just encode the entire cfg.MODEL

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        if not self.tensor_mode:
            return self._wrapped_model.inference(inputs)

        with mock_preprocess_image(self._wrapped_model):
            with self.roi_heads_patcher.mock_roi_heads(self.tensor_mode):
                results = self._wrapped_model.inference(inputs, do_postprocess=False)
        return tuple(results[0].flatten())


class Caffe2PanopticFPN(Caffe2Compatible, torch.nn.Module):
    def __init__(self, cfg, torch_model):
        super(Caffe2PanopticFPN, self).__init__()
        assert isinstance(torch_model, meta_arch.PanopticFPN)
        self._wrapped_model = patch_generalized_rcnn(torch_model)
        self.eval()
        set_caffe2_compatible_tensor_mode(self, True)

        self.roi_heads_patcher = ROIHeadsPatcher(cfg, self._wrapped_model.roi_heads)

    def get_tensors_input(self, batched_inputs):
        return convert_batched_inputs_to_c2_format(
            batched_inputs,
            self._wrapped_model.backbone.size_divisibility,
            self._wrapped_model.device,
        )

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"PanopticFPN")

        # Inference parameters:
        check_set_pb_arg(predict_net, "combine_on", "i", self._wrapped_model.combine_on)
        check_set_pb_arg(
            predict_net,
            "combine_overlap_threshold",
            "f",
            _cast_to_f32(self._wrapped_model.combine_overlap_threshold),
        )
        check_set_pb_arg(
            predict_net,
            "combine_stuff_area_limit",
            "i",
            self._wrapped_model.combine_stuff_area_limit,
        )
        check_set_pb_arg(
            predict_net,
            "combine_instances_confidence_threshold",
            "f",
            _cast_to_f32(self._wrapped_model.combine_instances_confidence_threshold),
        )

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        """
        Re-write the inference-only forward pass of PanopticFPN in c2 style
        """
        assert self.tensor_mode

        images = caffe2_preprocess_image(self._wrapped_model, inputs)
        features = self._wrapped_model.backbone(images.tensor)

        gt_sem_seg = None
        sem_seg_results, _ = self._wrapped_model.sem_seg_head(features, gt_sem_seg)
        sem_seg_results = alias(sem_seg_results, "sem_seg")

        gt_instances = None
        proposals, _ = self._wrapped_model.proposal_generator(
            images, features, gt_instances
        )

        with self.roi_heads_patcher.mock_roi_heads(self.tensor_mode):
            detector_results, _ = self._wrapped_model.roi_heads(
                images, features, proposals, gt_instances
            )

        return tuple(detector_results[0].flatten()) + (sem_seg_results,)


class Caffe2RetinaNet(Caffe2Compatible, torch.nn.Module):
    def __init__(self, cfg, torch_model):
        super(Caffe2RetinaNet, self).__init__()
        assert isinstance(torch_model, meta_arch.RetinaNet)
        self._wrapped_model = torch_model
        self.eval()
        set_caffe2_compatible_tensor_mode(self, True)

        # serialize anchor_generator for future use
        self._serialized_anchor_generator = io.BytesIO()
        torch.save(
            self._wrapped_model.anchor_generator, self._serialized_anchor_generator
        )

    def get_tensors_input(self, batched_inputs):
        return convert_batched_inputs_to_c2_format(
            batched_inputs,
            self._wrapped_model.backbone.size_divisibility,
            self._wrapped_model.device,
        )

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"RetinaNet")

        # Inference parameters:
        check_set_pb_arg(
            predict_net,
            "score_threshold",
            "f",
            _cast_to_f32(self._wrapped_model.score_threshold),
        )
        check_set_pb_arg(
            predict_net, "topk_candidates", "i", self._wrapped_model.topk_candidates
        )
        check_set_pb_arg(
            predict_net,
            "nms_threshold",
            "f",
            _cast_to_f32(self._wrapped_model.nms_threshold),
        )
        check_set_pb_arg(
            predict_net,
            "max_detections_per_image",
            "i",
            self._wrapped_model.max_detections_per_image,
        )

        check_set_pb_arg(
            predict_net,
            "bbox_reg_weights",
            "floats",
            [_cast_to_f32(w) for w in self._wrapped_model.box2box_transform.weights],
        )
        self._encode_anchor_generator_cfg(predict_net)

    def _encode_anchor_generator_cfg(self, predict_net):
        # Ideally we can put anchor generating inside the model, then we don't
        # need to store this information.
        bytes = self._serialized_anchor_generator.getvalue()
        check_set_pb_arg(predict_net, "serialized_anchor_generator", "s", bytes)

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        assert self.tensor_mode
        images = caffe2_preprocess_image(self._wrapped_model, inputs)

        # explicitly return the images sizes to avoid removing "im_info" by ONNX
        # since it's not used in the forward path
        return_tensors = [images.image_sizes]

        features = self._wrapped_model.backbone(images.tensor)
        features = [features[f] for f in self._wrapped_model.in_features]

        box_cls, box_delta = self._wrapped_model.head(features)
        for i, (box_cls_i, box_delta_i) in enumerate(zip(box_cls, box_delta)):
            return_tensors.append(alias(box_cls_i, "box_cls_{}".format(i)))
            return_tensors.append(alias(box_delta_i, "box_delta_{}".format(i)))

        return tuple(return_tensors)


META_ARCH_CAFFE2_EXPORT_TYPE_MAP = {
    "GeneralizedRCNN": Caffe2GeneralizedRCNN,
    "PanopticFPN": Caffe2PanopticFPN,
    "RetinaNet": Caffe2RetinaNet,
}


def _export_via_onnx(model, inputs):
    # make sure all modules are in eval mode, onnx may change the training state
    #  of the moodule if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                # verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization
    all_passes = onnx.optimizer.get_available_passes()
    passes = ["fuse_bn_into_conv"]
    assert all(p in all_passes for p in passes)
    onnx_model = onnx.optimizer.optimize(onnx_model, passes)

    # Convert ONNX model to Caffe2 protobuf
    init_net, predict_net = c2.onnx_graph_to_caffe2_net(onnx_model)

    return predict_net, init_net


def _op_stats(net_def):
    type_count = {}
    for t in [op.type for op in net_def.op]:
        type_count[t] = type_count.get(t, 0) + 1
    type_count_list = sorted(type_count.items(), key=lambda kv: kv[0])  # alphabet
    type_count_list = sorted(type_count_list, key=lambda kv: -kv[1])  # count
    return "\n".join("{:>4}x {}".format(count, name) for name, count in type_count_list)


def export_caffe2_detection_model(
    model: torch.nn.Module, tensor_inputs: List[torch.Tensor]
):
    """
    Export a Detectron2 model via ONNX.

    Arg:
        model: a caffe2-compatible version of detectron2 model, defined in caffe2_modeling.py
        tensor_inputs: a list of tensors that caffe2 model takes as input.
    """
    model = copy.deepcopy(model)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "encode_additional_info")

    # Export via ONNX
    logger.info("Exporting {} model via ONNX ...".format(type(model)))
    predict_net, init_net = _export_via_onnx(model, (tensor_inputs,))
    logger.info("ONNX export Done.")

    # Apply protobuf optimization
    fuse_alias_placeholder(predict_net, init_net)
    params = get_params_from_init_net(init_net)
    predict_net, params = remove_reshape_for_fc(predict_net, params)
    group_norm_replace_aten_with_caffe2(predict_net)
    init_net = construct_init_net_from_params(params)

    # Record necessary information for running the pb model in Detectron2 system.
    model.encode_additional_info(predict_net, init_net)

    logger.info("Operators used in predict_net: \n{}".format(_op_stats(predict_net)))
    logger.info("Operators used in init_net: \n{}".format(_op_stats(init_net)))

    return predict_net, init_net


def run_and_save_graph(predict_net, init_net, tensor_inputs, graph_save_path):
    """
    Run the caffe2 model on given inputs, recording the shape and draw the graph.

    predict_net/init_net: caffe2 model.
    tensor_inputs: a list of tensors that caffe2 model takes as input.
    graph_save_path: path for saving graph of exported model.
    """

    logger.info("Saving graph of ONNX exported model to {} ...".format(graph_save_path))
    save_graph(predict_net, graph_save_path, op_only=False)

    # Run the exported Caffe2 net
    logger.info("Running ONNX exported model ...")
    with ScopedWS("__ws_tmp__", True) as ws:
        ws.RunNetOnce(init_net)
        initialized_blobs = set(ws.Blobs())
        uninitialized = [
            inp for inp in predict_net.external_input if inp not in initialized_blobs
        ]
        for name, blob in zip(uninitialized, tensor_inputs):
            ws.FeedBlob(name, blob)

        try:
            ws.RunNetOnce(predict_net)
        except RuntimeError as e:
            logger.warning("Encountered RuntimeError: \n{}".format(str(e)))

        ws_blobs = {b: ws.FetchBlob(b) for b in ws.Blobs()}
        blob_sizes = {
            b: ws_blobs[b].shape
            for b in ws_blobs
            if is_valid_model_output_blob(ws_blobs[b])
        }

        logger.info("Saving graph with blob shapes to {} ...".format(graph_save_path))
        save_graph(predict_net, graph_save_path, op_only=False, blob_sizes=blob_sizes)

        return ws_blobs