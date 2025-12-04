#!/usr/bin/env python

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import time
from math import ceil
from statistics import mean, mode

import cv2
import numpy as np
import skvideo.io
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

# import tools
import stgcn_lrp.tools.utils as utils
from stgcn_lrp.tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new

from .io import IO
from .lrp_src.lrp import LRPModel
from .processor import Processor

max_body = 2
num_joint = 1
max_frame = 601

drop_stride = 2

class SkeletonLRP(Processor):

    def start(self):
        # print(self.model)
        print("Using {} weights.".format(self.arg.valid))
        out = self.process_lrp()

        return

    def cfg_hook(self):
        self.gradients, self.activations = [], []

        def backward_hook_fn(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        def forward_hook_fn(module, feat_in, feat_out):
            self.activations.append(feat_out.detach())

        def input_backward_hook_fn(module, feat_in, feat_out):
            self.input_grad = feat_out

        modules = list(self.model.modules())
        for module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
            if isinstance(module, torch.nn.Conv2d):
                module.register_backward_hook(input_backward_hook_fn)

        return

    def reset_hook_value(self):
        self.gradients, self.activations = [], []
        return

    def process_lrp(self):

        self.loss = torch.nn.CrossEntropyLoss()
        # self.model = self.model.double()
        # initiate
        # label from http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
        label_name_path = './resource/ntu_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        skeleton_name = self.arg.skeleton
        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
        print("Processing skeleton:", skeleton_name)

        skeleton_file = '../NTU-RGB-D-120/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = read_xyz_new(
            skeleton_file, max_body=max_body, num_joint=num_joint)

        self.cfg_hook()

        data = torch.from_numpy(data_numpy).float()
        data = data.unsqueeze(0)

        # saliency_map, model_output, pred_class = cam_func(data, torch.from_numpy(np.array([action_class])).unsqueeze(0).type(torch.int64))
        # self.evaluate(data, saliency_map, torch.from_numpy(np.array([action_class])).type(torch.long))
        for i in range(10):
            layer = str(i)
            output_result_dir = '{}/lrp_ori/{}'.format(self.arg.output_dir, skeleton_name)
            if not os.path.exists('{}'.format(output_result_dir)):
                os.makedirs('{}'.format(output_result_dir))

            saliency_map = self.lrp_func(data, layer)
            
            print(saliency_map.max(), saliency_map.min())
            print("Processing layer:", layer)
            
            

            utils.visualization_skeleton.plot_action(
                data_numpy, self.model.graph.edge, saliency_map[0].cpu().numpy(),
                save_dir=output_result_dir, save_type=layer)

        return

    def lrp_func(self, data, layer):
        self.model.eval()
        self.model.zero_grad()
        # print(self.model)
        lrp_model = LRPModel(model=self.model)
        r = lrp_model.forward(data, layer)
        return r

    def render_skeleton(self, data_numpy, intensity, voting_label_name, video_label_name=None):
        images = utils.visualization_skeleton.stgcn_visualize_3d(
            data_numpy,
            self.model.graph.edge,
            intensity,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def render_correlation(self, hidden_feature):
        return utils.visualization_skeleton.extract_correlation(hidden_feature)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--skeleton',
                            default='S001C001P001R001A006',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--plot_action',
                            default=True,
                            help='save action as image',
                            type=bool)
        parser.add_argument('--output_dir',
                            default='./data/lrp_skeleton/result',
                            help='Path to save results')
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--cam_type',
                            default='gradcam',
                            help='One of gradcam, gradcampp, smoothcam, \
                                ablation, scorecam, ada-gradcam, axiom, l2-caf')
        parser.add_argument('--data_level',
                            default='test_set',
                            help='instance or test_set')
        parser.add_argument('--valid',
                            default='xsub',
                            help='One of xsub and xview')
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

        args = parser.parse_known_args(namespace=parent_parser)
        # print("mark", parent_parser.valid)
        parser.set_defaults(
            config='./config/st_gcn/ntu-{}/demo_skeleton_cam.yaml'.format(parent_parser.valid))
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
