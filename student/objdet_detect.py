# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict
import os
import sys

# Import only necessary functions/classes from modules
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing
from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2

# Load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):
    if configs is None:
        configs = edict()
    configs.min_iou = 0.4

    if model_name == "darknet":
        configs.model_path = os.path.join('tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False
    elif model_name == 'fpn_resnet':
        configs.model_path = os.path.join('tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet'
        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_layers = 18
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2
        configs.K = 50
        configs.down_ratio = 4
        configs.conf_thresh = 0.2
        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
    else:
        raise ValueError("Error: Invalid model name")

    configs.no_cuda = True
    configs.gpu_idx = 0
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    return configs


# Load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):
    if configs is None:
        configs = edict()
    configs.lim_x = [0, 50]
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0]
    configs.bev_width = 608
    configs.bev_height = 608
    configs = load_configs_model(model_name, configs)
    configs.output_width = 608
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
    return configs


# Create model according to selected model type
def create_model(configs):
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)
    elif 'fpn_resnet' in configs.arch:
        model = fpn_resnet.get_pose_net(num_layers=configs.num_layers, heads=configs.heads,
                                        head_conv=configs.head_conv, imagenet_pretrained=configs.imagenet_pretrained)
    else:
        assert False, 'Undefined model backbone'

    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()
    return model


# Detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):
    with torch.no_grad():
        outputs = model(input_bev_maps)
        if 'darknet' in configs.arch:
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])
        elif 'fpn_resnet' in configs.arch:
            outputs['hm_cen'] = torch.sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = torch.sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)
            detections = detections[0][1]

    objects = []
    if len(detections) > 0:
        for det in detections:
            cls_id, _x, _y, _z, _h, _w, _l, _yaw = det
            _yaw = -_yaw
            x = _y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0]) + configs.lim_x[0]
            y = _x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0]) + configs.lim_y[0]
            z = _z + configs.lim_z[0]
            w = _w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
            l = _l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            objects.append([cls_id, x, y, z, _h, w, l, _yaw])

    return objects
