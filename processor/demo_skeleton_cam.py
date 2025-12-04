#!/usr/bin/env python

import os
import sys
import argparse
import json
import shutil
import time
from statistics import mean, mode
from math import ceil
import copy
import gc

import numpy as np
import torch
import torch.nn.functional as F
import skvideo.io
import cv2
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .io import IO
from .processor import Processor
import tools
import tools.utils as utils
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new
import copy
import datetime
import itertools
import math
import os
import argparse
import time

from .processor import Processor
from torch.nn.functional import cross_entropy
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new
from itertools import combinations, chain
# from keras.utils import to_categorical
import tools.utils as utils

import numpy as np
import torch

max_body = 2
num_joint = 25
max_frame = 300
unique_keys = ['left_hand', 'right_hand', 'left_leg', 'right_leg', 'trunk']
drop_stride = 2
#  记录文本
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
os.makedirs('2', exist_ok=True)
file_txt = open('2/' + time_now + '.txt', mode='a', encoding="utf-8")
num_devices = torch.cuda.device_count()


class DemoSkeletonCam(Processor):

    # 初始化并选择CAM方法
    def start(self):
        # self.dev = "cuda:0"
        self.dev = self.arg.run_device
        print(self.arg.run_device)
        print(self.arg.cam_type)
        # print(self.model)
        print("Using {} weights.".format(self.arg.valid))
        cams_dict = {'guided_bp': self.Guided_BP,
                     'gradcam': self.GradCam,
                     'gradcampp': self.GradCampp,
                     'smoothgradcampp': self.SmoothGradCampp,
                     'scorecam': self.ScoreCam,
                     'ablation': self.Ablation,
                     'integ': self.IntegratedGrad,
                     'axiom': self.Axiom,
                     'efcam': self.EFCAM,
                     'efcam_sm': self.EFCAM_softmax,
                     'inablasp': self.in_abla_softplus,
                     'efcam_mar': self.EFCAM_margin,
                     'efcam_mar_eff': self.EFCAM_margin_eff,
                     'efcam_mar_bs': self.EFCAM_margin_bs,
                     'uocam': self.UOCAM,
                     'bicam': self.BICAM_Appendix,
                     'imcam': self.IMCam,
                     'isgcam': self.ISGCam,
                     'shapleycam': self.ShapleyCam}
        self.method = self.arg.cam_type.lower()
        out = self.process_cams(cams_dict[self.method])

        return

    # 注册前向和反向钩子来捕获梯度和激活值
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

    def gen_mask(self, data):  # 此处可以再做优化，仅保存非零t和非零m
        '''
        Generate mask for data
        :param data: Original data with shape(bs, c, t, v, m)
        :return: non_zero mask with shape(bs, 1, t, 1, m)
        '''
        non_zero = (data != 0).type(torch.uint8)

        non_zero_btm = (non_zero.sum(dim=(1, 3), keepdim=False) != 0)
        non_zero_t = (non_zero_btm.sum(dim=2, keepdim=False) != 0).sum(-1)
        num_zero_m = (non_zero_btm.sum(dim=1, keepdim=False) != 0).sum(-1)
        mask = torch.stack((non_zero_t, num_zero_m), dim=1)

        return mask

    def summary(self, ave_drop, incr, inse_auc, del_auc):
        if self.count == 0:
            self.ave_drop = ave_drop
            self.increase = incr
            self.inse_auc = inse_auc
            self.del_auc = del_auc

        else:
            self.ave_drop += ave_drop
            self.increase += incr
            self.inse_auc += inse_auc
            self.del_auc += del_auc

        self.count += 1

        if self.count % 50 == 0:
            # print("Cross Entropy:\n", self.loss_value)
            # for i in range(len(self.topk)):
            #     print("Top", self.topk[i], ":", self.top_k[:, i],
            #           "Batch size:", self.batch_size,
            #           "Count:", self.count,
            #           "Data count:", self.data_set_len)
            print("Average Drop:", self.ave_drop,
                  ", Increase:", self.increase,
                  ", Insersion AUC:", self.inse_auc,
                  ", Deletion AUC:", self.del_auc,
                  ", Data count:", self.count)
            file_txt.write("================================END==============================" + '\n')
            file_txt.write("总的平均下降指标：" + str(self.ave_drop) + '\n')
            file_txt.write("总的平均上升指标：" + str(self.increase) + '\n')
            file_txt.write("总的删除曲线指标：" + str(self.del_auc) + '\n')
            file_txt.write("总的插入曲线指标：" + str(self.inse_auc) + '\n')
            file_txt.write("轮次：" + str(self.count) + '\n')
            file_txt.write("================================END==============================" + '\n')
            # print()
        if self.count % 500 == 0:
            # print("Cross Entropy:\n", self.loss_value)
            # for i in range(len(self.topk)):
            #     print("Top", self.topk[i], ":", self.top_k[:, i],
            #           "Batch size:", self.batch_size,
            #           "Count:", self.count,
            #           "Data count:", self.data_set_len)
            print("Average Drop:", self.ave_drop / self.count,
                  ", Increase:", self.increase / self.count,
                  ", Insersion AUC:", self.inse_auc / self.count,
                  ", Deletion AUC:", self.del_auc / self.count,
                  ", Data count:", self.count,
                  ", cam:", self.arg.cam_type,
                  ", Data count:", self.arg.valid,
                  ",五个联盟，基于gradcampp")

    def process_cams(self, cam_func):

        self.loss = torch.nn.CrossEntropyLoss()
        # self.model = self.model.double()

        if self.arg.data_level == 'instance':
            print("---------------------------------------------------")
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

            output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_name)
            print(output_result_dir)
            output_cam_path = '{}/cam/{}'.format(output_result_dir, skeleton_name)
            if not os.path.exists('{}/cam/'.format(output_result_dir)):
                os.makedirs('{}/cam/'.format(output_result_dir))

            # skeleton_file = './data/NTU-RGB-D/nturgb+d_skeletons/'
            skeleton_file = './data/NTU-RGB-D/ntuall/'
            skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
            data_numpy = read_xyz(
                skeleton_file, max_body=max_body, num_joint=num_joint)

            self.cfg_hook()

            data = torch.from_numpy(data_numpy).float()
            data = data.unsqueeze(0)
            data = data.float().to(self.dev)
            data_model = data.to(self.dev)
            self.model = self.model.float()
            self.model = self.model.to(self.dev)
            out = self.model(data_model)
            label = out.argmax(dim=1, keepdim=True)
            # saliency_map, model_output, pred_class = cam_func(data, torch.from_numpy(np.array([action_class])).unsqueeze(0).type(torch.int64))
            # self.evaluate(data, saliency_map, torch.from_numpy(np.array([action_class])).type(torch.long))
            if self.method == "bicam":
                saliency_map, model_output, pred_class = cam_func(data, alpha=self.arg.alpha, k=self.arg.m,
                                                                  fix_ret=self.arg.fix_ret)
            elif self.method == "shapleycam":
                saliency_map = cam_func(data, label)
            else:
                saliency_map, model_output, pred_class = cam_func(data)

            self.evaluate(data, saliency_map)

            if self.arg.plot_action:
                utils.visualization_skeleton.plot_action(
                    data_numpy, self.model.graph.edge, saliency_map[0].cpu().numpy(),
                    save_dir=output_result_dir, save_type=self.arg.cam_type)

            return saliency_map
        # elif self.arg.data_level == 'test_set':
        #     loader = self.data_loader['test']
        #     print("-----------------------------------------")
        #     print(loader)
        #     self.cfg_hook()
        #     self.count = 0
        #     self.data_set_len = 0

        #     if self.method == "bicam":
        #         print("alpha :", self.arg.alpha, ", m :", self.arg.m, ", fix_ret :", self.arg.fix_ret)

        #     for data, label in loader:
        #         print(np.shape(data))
        #         print(label)
        #         self.model = self.model.float()
        #         self.model = self.model.to(self.dev)
        #         self.mask = self.gen_mask(data)
        #         data = data.float().to(self.dev)
        #         data_model = data.to(self.dev)
        #         with torch.no_grad():
        #             out = self.model(data_model)
        #             label_1 = out.argmax(dim=1, keepdim=True)
        #         # saliency_map, model_output, pred_class = cam_func(data, label)
        #         if self.method == "bicam":
        #             saliency_map, model_output, pred_class = cam_func(data, alpha=self.arg.alpha, k=self.arg.m,
        #                                                               fix_ret=self.arg.fix_ret)
        #         elif self.method =="shapleycam":
        #             saliency_map = cam_func(data,label_1)
        #         else:
        #             saliency_map, model_output, pred_class = cam_func(data)
        #         ave_drop, incr, inse_auc, del_auc = self.evaluate(data, saliency_map, label)
        #         self.data_set_len += data.size(0)
        #         self.summary(ave_drop, incr, inse_auc, del_auc)
        #         if self.data_set_len > 2005:
        #             return

        #     self.label_list = self.label_list.cpu()
        #     self.drop_class_list = self.drop_class_list.cpu()
        #     acc_score_list = []
        #     f1_score_list = []
        #     recall_score_list = []
        #     for i in range(self.drop_class_list.size(1)):
        #         acc_score_list.append(accuracy_score(self.label_list, self.drop_class_list[:, i]))
        #         f1_score_list.append(f1_score(self.label_list, self.drop_class_list[:, i], average='micro'))
        #         recall_score_list.append(recall_score(self.label_list, self.drop_class_list[:, i], average='micro'))
        #     print("Acc Score:", acc_score_list,
        #           "\nF1 Score:", f1_score_list,
        #           "\nRecall Score:", recall_score_list)

        #     self.loss_value /= self.data_set_len
        #     self.ave_drop /= self.data_set_len
        #     print("Global Mean Cross Entropy:\n", self.loss_value)

        #     for i in range(len(self.arg.topk)):
        #         print("Top", self.arg.topk[i], ":", self.top_k[:, i],
        #               "Batch size:", self.arg.test_batch_size,
        #               "Data Count:", self.data_set_len)

        #     print("Global Average Drop:", self.ave_drop)
        #     print("Global Increase Num:", self.increase, "\nGlobal Data Num:", self.count)
        #     self.increase = self.increase.astype(float)
        #     self.increase /= self.data_set_len
        #     print("Global Increase Pro:", self.increase)
        #     return
        # else:
        #     assert False, "Aug test_set must be one of instance and test_set."

        elif self.arg.data_level == 'test_set':
            loader = self.data_loader['test']
            self.cfg_hook()
            self.count = 0
            self.data_set_len = 0
            count = 0

            fidelity_acc_out = 0
            infidelity_acc_out = 0
            fidelity_prob_out = 0
            infidelity_prob_out = 0
            Sparsity_out = 0

            if self.method == "bicam":
                print("alpha :", self.arg.alpha, ", m :", self.arg.m, ", fix_ret :", self.arg.fix_ret)

            # for data, label in loader:
            skeleton_file_path = './data/NTU-RGB-D/nturgb+d_skeletons/'
            # skeleton_file_path = './data/NTU-RGB-D/ntuall/'
            for root, dirs, files in os.walk(skeleton_file_path):

                for file in files:
                    file_txt.write("=================================================================" + '\n')
                    # file_txt.write(str(datetime.datetime.now()) + '\n')
                    file_txt.write(str(self.arg.cam_type.lower()) + '\n')
                    file_txt.write(str(file) + '\n')
                    file_txt.write(str(self.arg.cam_type.lower()) + '\n')
                    file_txt.write(str(self.arg.valid) + '\n')
                    file_txt.write(str(self.arg.config) + '\n')
                    skeleton_file = skeleton_file_path + file
                    data_numpy = read_xyz(skeleton_file, max_body=max_body, num_joint=num_joint)
                    self.model.eval()
                    self.cfg_hook()
                    print(file)
                    data = torch.from_numpy(data_numpy)
                    data = data.unsqueeze(0)
                    data = data.float().to(self.dev)
                    data_model = data.to(self.dev)
                    self.model = self.model.float()
                    self.model = self.model.to(self.dev)
                    with torch.no_grad():
                        out = self.model(data_model)
                        label = out.argmax(dim=1, keepdim=True)
                        probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
                    print("Processing skeleton:", file)
                    file_txt.write("Processing skeleton:" + file)
                    # saliency_map, model_output, pred_class = cam_func(data, label)

                    if self.method == "bicam":
                        saliency_map, model_output, pred_class = cam_func(data, alpha=self.arg.alpha, k=self.arg.m,
                                                                          fix_ret=self.arg.fix_ret)
                    elif self.method == "shapleycam":
                        saliency_map = cam_func(data, label)
                    else:
                        saliency_map, model_output, pred_class = cam_func(data)
                    file_txt.write(str(saliency_map) + '\n')
                    output_result_dir = '{}/{}'.format(self.arg.output_dir, file)

                    fidelity_acc, infidelity_acc, fidelity_prob, infidelity_prob, Sparsity = \
                        self.evaluate_1(saliency_map, data, label, probability)

                    fidelity_acc_out += fidelity_acc
                    infidelity_acc_out += infidelity_acc
                    fidelity_prob_out += fidelity_prob
                    infidelity_prob_out += infidelity_prob
                    Sparsity_out += Sparsity
                    count = count + 1
                    # self.summary(ave_drop, incr, inse_auc, del_auc)
                    print('count', count)
                    if count >= 2000:
                        break

            fidelity_acc = fidelity_acc_out / count
            infidelity_acc = infidelity_acc_out / count
            fidelity_prob = fidelity_prob_out / count
            infidelity_prob = infidelity_prob_out / count
            Sparsity = Sparsity_out / count

            print("=================================================================")

            print("评价指标：指标忠诚度")

            print("fidelity_acc:", format(fidelity_acc, '.3f'))

            print("infidelity_acc:", format(infidelity_acc, '.3f'))

            print("fidelity_prob:", format(fidelity_prob, '.3f'))

            print("infidelity_prob:", format(infidelity_prob, '.3f'))

            print("Sparsity:", format(Sparsity, '.3f'))
            print("method:", format(self.arg.cam_type.lower()))
            print(self.arg.valid)
            return
        else:
            assert False, "Aug test_set must be one of instance and test_set."

    # def lead_data(self):
    #     skeleton_name = self.arg.skeleton
    #     action_class = int(
    #         skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
    #     print("Processing skeleton:", skeleton_name)
    #     print("Skeleton class:", action_class)
    #     self.action_class = action_class

    #     # skeleton_file = 'D:\\s\\st-gcn\\data\\NTU-RGB-D\\nturgb+d_skeletons\\'
    #     # skeleton_file = './data/NTU-RGB-D/nturgb+d_skeletons/'
    #     skeleton_file = './data/NTU-RGB-D/ntuall/'
    #     skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
    #     data_numpy = read_xyz(
    #         skeleton_file, max_body=max_body, num_joint=num_joint)

    #     self.model.eval().to(self.dev)
    #     data = torch.from_numpy(data_numpy)
    #     data = data.unsqueeze(0)
    #     data = data.float()
    #     data_model = data.to(self.dev)
    #     with torch.no_grad():
    #         out = self.model(data_model)
    #         label = out.argmax(dim=1, keepdim=True)
    #         probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
    #     print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    #     print(label)
    #     return data_numpy, data,  label, probability, skeleton_name

    # 计算忠诚度指标
    def evaluate_1(self, node_feature_mask, data, label, probability):
        return self.fidelity(node_feature_mask, data, label, probability)

    def data_reshape_1(self, data, mask):
        N, C, T, V, M = data.size()
        reshaped_mask = torch.reshape(mask, [N, T, V, M, C])
        reshaped_mask = reshaped_mask.permute(0, 4, 1, 2, 3).contiguous()

        return reshaped_mask

    def fidelity(self, node_feature_mask, data, label, probability):

        # <editor-fold desc="获得初始数据全1的mask">
        All_one_mask = torch.ones_like(node_feature_mask).to(self.dev)
        print(data.shape)

        node_mask = node_feature_mask.to(self.dev)
        un_node_mask = (All_one_mask - node_mask).to(self.dev)
        # </editor-fold>

        # <editor-fold desc="">
        print("模型识别的类别：", label, "   类别概率：", format(probability, '.4f'))

        # 计算fidelity+ acc 和 fidelity+ prob
        self.model.float().eval()

        with torch.no_grad():
            # 验证
            print("Input dtype:", (data * node_mask).float().to(self.dev).dtype)
            print("Model dtype:", next(self.model.parameters()).dtype)

            # 使用节点重要掩码
            importance_node_mask = (data * node_mask).float().to(self.dev)
            node_mask_out = self.model(importance_node_mask).to(self.dev)

            # 记录概率和标签
            node_mask_label = node_mask_out.argmax(dim=1, keepdim=True).item()
            node_mask_prob = torch.nn.functional.softmax(node_mask_out[0], dim=-1)[label].item()
            print("使用重要节点特征掩码类别：", node_mask_label, "     使用重要节点特征掩码概率：",
                  format(node_mask_prob, '.4f'))

            # 使用节点不重要掩码
            unimportance_node_mask = (data * un_node_mask).float().to(self.dev)
            un_node_mask_out = self.model(unimportance_node_mask).to(self.dev)
            # 记录概率和标签
            un_node_mask_label = un_node_mask_out.argmax(dim=1, keepdim=True).item()
            un_node_mask_prob = torch.nn.functional.softmax(un_node_mask_out[0], dim=-1)[label].item()
            print("使用不重要节点特征掩码类别：", un_node_mask_label, "     使用不重要节点特征掩码概率：",
                  format(un_node_mask_prob, '.4f'))

        # 开始计算Fidelity_ACC
        acc_node = 1 if label == node_mask_label else 0
        acc_un_node = 1 if label == un_node_mask_label else 0

        # 开始计算prob
        prob_node = probability - node_mask_prob
        prob_un_node = probability - un_node_mask_prob

        # 重要节点特征mask的正确率
        fidelity_acc = 1 - acc_un_node
        print("fidelity_acc:", fidelity_acc)

        # 不重要节点特征的mask的正确率
        infidelity_acc = 1 - acc_node
        print("infidelity_acc:", infidelity_acc)

        # 只有不重要帧mask的概率
        fidelity_prob = prob_un_node
        print("fidelity_prob:", format(fidelity_prob, '.4f'))

        # 所有不重要帧mask和节点特征的mask的正确率
        infidelity_prob = prob_node
        print("infidelity_prob:", format(infidelity_prob, '.4f'))

        Sparsity = 1 - (torch.sum(node_mask) / torch.sum(All_one_mask).item())

        print("Sparsity：", format(Sparsity, '.4f'))
        print("method:", format(self.arg.cam_type.lower()))
        print(self.arg.valid)

        return fidelity_acc, infidelity_acc, fidelity_prob, infidelity_prob, Sparsity

    def Guided_BP(self, data, action_class=None):
        '''Guided BackPropagation 方法'''
        # TODO: 这里的guide方法是错的，需要重新调整
        # 预处理，获取激活和梯度
        # inference
        self.model.eval()
        self.model.zero_grad()

        # action recognition
        data = data.float().to(self.dev).requires_grad_()

        model_output = self.model(data)
        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros(model_output.shape, dtype=torch.float)
        one_hot = one_hot.scatter_(1, action_class, 1)

        loss = torch.sum(one_hot * model_output)
        loss.backward(retain_graph=True)

        # 获取梯度
        input_grad = self.input_grad[-1].detach()
        self.reset_hook_value()

        input_grad = input_grad.norm(dim=1, keepdim=True)
        bs, c, t, v = input_grad.shape  # 实际处理中bs和m已经合并
        input_grad = input_grad.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        # bs, c, t, v, m = input_grad.shape

        # 核心处理方法
        saliency_map = F.relu(input_grad)

        return saliency_map.detach(), model_output.detach(), action_class.detach()

    def GradCam(self, data, action_class=None):
        '''GradCAM 方法'''
        if self.arg.data_level == "test_set":
            assert self.arg.test_batch_size == 1, "Test batch size for GradCAM method must be one."
        # 预处理，获取激活和梯度
        # inference
        self.model.eval().to(self.dev)
        self.model.zero_grad()

        # action recognition
        data = data.to(self.dev)

        model_output = self.model(data)

        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)

        loss = torch.sum(one_hot * model_output)
        loss.backward(retain_graph=True)

        # 获取梯度
        gradients = self.gradients[0].detach()
        activations = self.activations[-1].detach()
        self.reset_hook_value()

        bs, c, t, v = gradients.shape  # 实际处理中bs和m已经合并
        gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = gradients.shape

        # 核心处理方法
        alpha = gradients.view(bs, c, -1).mean(2)
        weights = alpha.view(bs, c, 1, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map).detach()
        print(saliency_map.shape)

        saliency_map = self.reshape_saliency_map(saliency_map, data)
        print(saliency_map.shape)

        return saliency_map, model_output.detach(), action_class.detach()

    def GradCampp(self, data, action_class=None):
        ''' 预处理，获取激活和梯度'''
        if self.arg.data_level == "test_set":
            assert self.arg.test_batch_size == 1, "Test batch size for GradCAM++ method must be one."
        # inference
        self.model = self.model.double()
        self.model.eval().to(self.dev)
        self.model.zero_grad()

        # action recognition
        data = data.double().to(self.dev)
        model_output = self.model(data)
        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)

        loss = torch.sum(one_hot * model_output)
        loss.backward(retain_graph=True)

        # 获取梯度
        gradients = self.gradients[0].detach()
        activations = self.activations[-1].detach()
        self.reset_hook_value()

        bs, c, t, v = gradients.shape  # 实际处理中bs和m已经合并
        gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = gradients.shape

        '''核心处理方法'''
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * alpha_num + \
                      activations.mul(gradients.pow(3)).sum(dim=(2, 3, 4), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-6)

        relu_grad = F.relu((model_output * one_hot).sum(dim=1, keepdim=False).exp() * gradients)
        weights = (alpha * relu_grad).view(bs, c, -1).sum(-1).view(bs, c, 1, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map).detach()
        saliency_map = self.reshape_saliency_map(saliency_map, data)

        return saliency_map, model_output.detach(), action_class.detach()

    def SmoothGradCampp(self, data, action_class=None, n_samples=5, stdev_spread=0.10):
        # TODO: 这里的误差太大，已经干扰到结果
        if self.arg.data_level == "test_set":
            assert self.arg.test_batch_size == 1, "Test batch size for SmoothGradCAM++ method must be one."

        stdev = stdev_spread / (data.max() - data.min())
        std_tensor = torch.ones_like(data) * stdev
        indices = torch.zeros((self.arg.test_batch_size, n_samples), dtype=torch.uint8)

        for i in range(n_samples):
            self.model.eval()
            self.model.zero_grad()

            # 输入图片增加高斯噪声
            data_with_noise = torch.normal(mean=data, std=std_tensor)

            # action recognition
            data_with_noise = data_with_noise.float().to(self.dev)
            model_output = self.model(data_with_noise)
            if action_class is None:
                action_class = model_output.argmax(dim=1, keepdim=True)

            if i == 0:
                output = model_output.detach()
            else:
                output += model_output.detach()

            indices[:, i:i + 1] = action_class.detach()
            one_hot = torch.zeros_like(model_output)
            one_hot = one_hot.scatter_(1, action_class, 1)

            loss = torch.sum(one_hot * model_output)
            loss.backward(retain_graph=True)

            # 获取梯度
            gradients = self.gradients[0].detach()
            activations = self.activations[-1].detach()
            self.reset_hook_value()

            bs, c, t, v = gradients.shape  # 实际处理中bs和m已经合并
            gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
            activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
            bs, c, t, v, m = gradients.shape

            '''核心处理方法'''
            alpha_num = gradients.pow(2)
            alpha_denom = 2 * alpha_num + \
                          activations.mul(gradients.pow(3)).sum(axis=(2, 3, 4), keepdim=True)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alpha = alpha_num.div(alpha_denom)
            relu_grad = F.relu((model_output * one_hot).sum(dim=1, keepdim=False).exp() * gradients)
            weights = (alpha * relu_grad).view(bs, c, -1).sum(-1).view(bs, c, 1, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)

            if i == 0:
                total_map = saliency_map.detach()
            else:
                total_map += saliency_map.detach()

        total_map /= n_samples
        output /= n_samples
        index = indices.mode(dim=1, keepdim=False).values

        total_map = self.reshape_saliency_map(total_map, data)

        return total_map.detach(), output.detach(), index.detach()

    def ScoreCam(self, data, action_class=None):

        self.model.eval().to(self.dev)
        self.model.zero_grad()
        # action recognition
        data = data.to(self.dev)
        model_output = self.model(data)
        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)
        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).double().to(self.dev)
        score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev)

        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map,
                                             size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()

                output = F.softmax(output, dim=1)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1, 1, 1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重

                score_saliency_map += (score * saliency_map).detach()

        score_saliency_map = self.data_norm(F.relu(score_saliency_map))
        # score_saliency_map 为所求
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def Ablation(self, data, action_class=None):
        # TODO：Ablation方法可以采用非简化方式，需要参考官方程序
        self.model.eval().to(self.dev)
        self.model.zero_grad()

        # action recognition
        data = data.to(self.dev)
        model_output = self.model(data)
        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros(model_output.shape, dtype=torch.float).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1, 1, 1, 1).contiguous()
        # TODO: 没有经过softmax的socre是否准确，需要对照原论文和源程序以及论文原始模型进行思考和调整

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # slope = torch.zeros(bs, c, 1, 1, 1).double().to(self.dev)
        slope = torch.zeros(bs, c, 1, 1, 1).to(self.dev)

        with torch.no_grad():
            for i in range(c):
                self.model.eval()
                self.model.zero_grad()

                ablation_activations = activations.clone().detach()
                ablation_activations = ablation_activations.permute(0, 4, 1, 2, 3).contiguous()
                ablation_activations = ablation_activations.view(bs * m, c, t, v).contiguous()
                ablation_activations[:, i:i + 1, :, :] = torch.zeros(bs * m, 1, t, v)

                ablation_activations = F.avg_pool2d(ablation_activations, ablation_activations.size()[2:])
                ablation_activations = ablation_activations.view(bs, m, -1, 1, 1).mean(dim=1)

                # prediction
                ablation_activations = self.model.fcn(ablation_activations)
                ablation_output = ablation_activations.view(ablation_activations.size(0), -1)
                ablation_score = torch.sum(one_hot * ablation_output, dim=1).view(-1, 1, 1, 1, 1).contiguous()
                self.reset_hook_value()

                slope[:, i:i + 1, :, :, :] = (model_score - ablation_score) / model_score

        saliency_map = (slope * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)

        saliency_map = self.reshape_saliency_map(saliency_map, data)

        return saliency_map.detach(), model_output.detach(), action_class.detach()

    def IntegratedGrad(self, data, action_class=None, num_steps=50):
        if self.arg.data_level == "test_set":
            assert self.arg.test_batch_size == 1, "Test batch size for Integrated Grad method must be one."

        baseline = torch.zeros(data.shape)
        bs, c, t, v, m = data.shape

        interpolated_data = [
            baseline + (step / num_steps) * (data - baseline)
            for step in range(num_steps + 1)]

        interpolated_grad = torch.zeros(num_steps + 1, bs, 192, t, v, m)

        for i, inter_data in enumerate(interpolated_data):
            self.model.eval()
            self.model.zero_grad()

            # action recognition
            inter_data = inter_data.float().to(self.dev).requires_grad_()  # (1, channel, frame, joint, person)

            model_output = self.model(inter_data)
            if not action_class:
                action_class = model_output.argmax(dim=1, keepdim=True)

            one_hot = torch.zeros(model_output.shape, dtype=torch.float)
            one_hot[0][action_class] = 1

            loss = torch.sum(one_hot * model_output)
            loss.backward(retain_graph=True)

            input_grad = self.input_grad[-1].detach()
            bs, c, t, v = input_grad.shape  # 实际处理中bs和m已经合并
            input_grad = input_grad.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
            interpolated_grad[i] = input_grad

        interpolated_grad = (interpolated_grad[:-1] + interpolated_grad[1:]) / 2.0
        avg_grad = interpolated_grad.mean(dim=0, keepdim=False)

        saliency_map = (data - baseline) * avg_grad.mean(dim=1, keepdim=True)
        saliency_map = saliency_map.mean(dim=1, keepdim=True)

        saliency_map = self.reshape_saliency_map(saliency_map, data)

        return saliency_map, model_output, action_class

    def Axiom(self, data, action_class=None):
        '''Axiom 方法'''
        if self.arg.data_level == "test_set":
            assert self.arg.test_batch_size == 1, "Test batch size for Axiom method must be one."
        # 预处理，获取激活和梯度
        # inference
        self.model.eval()
        self.model.zero_grad()

        # action recognition
        data = data.float().to(self.dev)

        model_output = self.model(data)
        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros(model_output.shape, dtype=torch.float).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)

        loss = torch.sum(one_hot * model_output)
        loss.backward(retain_graph=True)

        # 获取梯度
        gradients = self.gradients[0].detach()
        activations = self.activations[-1].detach()
        bs, c, t, v = gradients.shape  # 实际处理中bs和m已经合并

        gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()

        # 核心处理方法

        x_weights = (gradients * activations).sum(dim=(2, 3, 4), keepdim=True)
        x_weights = x_weights / (activations.sum(dim=(2, 3, 4), keepdim=True) + 1e-6)

        saliency_map = (x_weights * activations).sum(dim=1, keepdim=True)
        saliency_map = F.relu(saliency_map)

        saliency_map = self.reshape_saliency_map(saliency_map, data)

        return saliency_map, model_output, action_class

    def EFCAM(self, data, action_class=None):  # Erasing Feature CAM
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.float().to(self.dev)
        model_output = self.model(data)
        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1, 1, 1, 1).contiguous()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()

        slope = torch.zeros(bs, c, 1, 1, 1).to(self.dev)
        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()
                output = F.softmax(output, dim=1)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1, 1, 1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重
                slope[:, i:i + 1, :, :, :] = (model_score - score) / model_score

        # relu去除负值
        # score_saliency_map = F.relu(score_saliency_map)
        saliency_map = (slope * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        # 归一化
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def EFCAM_softmax(self, data, action_class=None):  # Erasing Feature CAM
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.float().to(self.dev)
        model_output = self.model(data)
        # model_output = F.softmax(model_output, dim=-1)
        # model_output = F.softplus(model_output)
        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1, 1, 1, 1).contiguous()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()

        slope = torch.zeros(bs, c, 1, 1, 1).to(self.dev)
        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()

                # output = F.softmax(output, dim=-1)
                # output = F.softplus(output)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1, 1, 1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重
                slope[:, i:i + 1, :, :, :] = (model_score - score)

        # relu去除负值
        slope = F.softmax(slope, dim=1)
        saliency_map = (slope * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        # 归一化
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def EFCAM_margin(self, data, action_class=None):  # Erasing Feature CAM
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.float().to(self.dev)
        model_output = self.model(data)
        # TODO：精简化程序margin部分，支持batchsize大于一，支持多卡

        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1).contiguous()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()

        slope = torch.zeros(bs, c).to(self.dev)
        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()

                # output = F.softmax(output, dim=1)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重
                slope[:, i:i + 1] = model_score - score  # score 越小代表覆盖区域越精准

        sorted_slope, indices_slope = torch.sort(slope, dim=-1, descending=True)
        slope_s = torch.zeros(bs, c).to(self.dev)
        slope_s[:, indices_slope[:, 0]] = slope[:, indices_slope[:, 0]]
        # TODO 这里有问题
        with torch.no_grad():
            saliency_map_a = activations[:, indices_slope[:, 0], :, :, :].detach()
            norm_saliency_map_a = self.data_norm(saliency_map_a)
            for i in range(1, c):
                saliency_map_b = activations[:, indices_slope[:, i], :, :, :].detach()
                # 归一化
                norm_saliency_map_b = self.data_norm(saliency_map_b)
                norm_saliency_map = 1 - self.data_norm(norm_saliency_map_a * norm_saliency_map_b)
                norm_saliency_map = F.interpolate(norm_saliency_map,
                                                  size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()

                score_s = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                slope_s[:, indices_slope[:, i]] = score_s
                if score_s < slope_s[:, indices_slope[:, i - 1]]:  # TODO:这里也有问题
                    slope_s[:, indices_slope[:, i]] = slope_s[:, indices_slope[:, i - 1]]
                else:
                    slope_s[:, indices_slope[:, i]] = score_s
                    norm_saliency_map_a = self.data_norm(norm_saliency_map_a * norm_saliency_map_b)

        # 计算排序后的边缘值
        slope_margin_percent = torch.ones_like(slope_s)
        slope_margin_percent[:, indices_slope[:, 0]] = slope_s[:, indices_slope[:, 0]]
        for i in range(1, c - 1):
            slope_margin_percent[:, indices_slope[:, i]] = (slope_s[:, indices_slope[:, i]] - slope_s[:,
                                                                                              indices_slope[:,
                                                                                              i - 1]])  # / slope_s[:, indices_slope[:, i]]  # TODO: 这里程序实现也有问题
        # 计算边缘值所占百分比
        slope_margin_percent[:, indices_slope[:, 0]] = 1
        for i in range(1, c - 1):
            slope_margin_percent[:, indices_slope[:, :i]] *= 1 - slope_margin_percent[:, indices_slope[:, i]]

        # relu去除负值
        slope_margin_percent = F.softmax(slope_margin_percent, dim=-1)
        slope_margin_percent = slope_margin_percent.view(bs, c, 1, 1, 1)
        saliency_map = (slope_margin_percent * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        # 归一化
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def EFCAM_margin_wrong(self, data, action_class=None):  # Erasing Feature CAM
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.float().to(self.dev)
        model_output = self.model(data)
        # TODO：精简化程序margin部分，支持batchsize大于一，支持多卡

        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1).contiguous()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape

        slope = torch.zeros(bs, c).to(self.dev)
        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()

                # output = F.softmax(output, dim=1)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重
                slope[:, i:i + 1] = model_score - score  # score 越小代表覆盖区域越精准

        sorted_slope, indices_slope = torch.sort(slope, dim=-1, descending=True)
        sub_slope = torch.zeros(bs, c).to(self.dev)
        sub_slope[:, 0] = slope[:, 0]  # TODO:这里程序实现也有问题

        with torch.no_grad():
            saliency_map_a = activations[:, indices_slope[:, 0], :, :, :].detach()
            norm_saliency_map_a = self.data_norm(saliency_map_a)
            for i in range(1, c):
                saliency_map_b = activations[:, indices_slope[:, i], :, :, :].detach()
                # 归一化
                norm_saliency_map_b = self.data_norm(saliency_map_b)
                norm_saliency_map = 1 - self.data_norm(norm_saliency_map_a * norm_saliency_map_b)
                norm_saliency_map = F.interpolate(norm_saliency_map,
                                                  size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()

                sub_score = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                sub_slope[:, indices_slope[:, i]] = sub_score
                if sub_score < slope[:, indices_slope[:, i]]:  # TODO: 这里程序也有问题
                    sub_slope[:, indices_slope[:, i]] = sub_slope[:, indices_slope[:, i - 1]]
                else:
                    sub_slope[:, indices_slope[:, i]] = sub_score
                    norm_saliency_map_a = self.data_norm(norm_saliency_map_a * norm_saliency_map_b)

        # 计算排序后的边缘值
        slope_margin_percent = torch.ones_like(sub_slope)
        for i in range(1, c - 1):
            slope_margin_percent[:, indices_slope[:, i]] = (sub_slope[:, indices_slope[:, i]] - slope[:,
                                                                                                indices_slope[:,
                                                                                                i - 1]]) \
                                                           / sub_slope[:, indices_slope[:, i]]  # TODO：这里程序实现有问题

        # 计算边缘值所占百分比
        for i in range(1, c - 1):
            slope_margin_percent[:, indices_slope[:, :i]] *= 1 - slope_margin_percent[:, indices_slope[:, i]]

        # relu去除负值
        slope_margin_percent = F.softmax(slope_margin_percent, dim=-1)
        slope_margin_percent = slope_margin_percent.view(bs, c, 1, 1, 1)
        saliency_map = (slope_margin_percent * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        # 归一化
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def EFCAM_margin_eff(self, data, action_class=None):  # Erasing Feature CAM
        # 这里的每一个部分都是使用了softmax的，否则top1和top5很高但是increase一直为0，混淆矩阵尚未测试
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.float().to(self.dev)
        model_output = self.model(data)
        # TODO：精简化程序margin部分，支持batchsize大于一，支持多卡

        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        model_output = F.softmax(model_output, dim=-1)
        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1).contiguous()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()

        slope = torch.zeros(bs, c).to(self.dev)
        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()
                output = F.softmax(output, dim=-1)

                # output = F.softmax(output, dim=1)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重
                slope[:, i:i + 1] = model_score - score  # score 越小代表覆盖区域越精准

        slope_s = torch.zeros(bs, c).to(self.dev)
        slope_s[:, 0] = slope[:, 0]
        sorted_slope, indices_slope = torch.sort(slope, dim=-1, descending=True)
        # print(sorted_slope, indices_slope)
        with torch.no_grad():
            # saliency_map_a = activations[:, indices_slope[:, 0], :, :, :].detach()
            norm_saliency_map_a = torch.ones_like(activations[:, 0, :, :])
            for i in range(c):
                saliency_map_b = activations[:, indices_slope[:, i], :, :, :].detach()
                # 归一化
                norm_saliency_map_b = self.data_norm(saliency_map_b)
                norm_saliency_map = 1 - self.data_norm(norm_saliency_map_a * norm_saliency_map_b)
                norm_saliency_map = F.interpolate(norm_saliency_map,
                                                  size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()
                output = F.softmax(output, dim=-1)

                score_s = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                slope_s[:, indices_slope[:, i]] = score_s
                norm_saliency_map_a = self.data_norm(norm_saliency_map_a * norm_saliency_map_b)
                # 计算排序后的边缘值
                # print("slope_s", slope_s)
            slope_margin_percent = torch.zeros_like(slope_s)
            slope_margin_percent[:, indices_slope[:, 0]] = (model_score[:] - slope_s[:, indices_slope[:, 0]] + 1e-6) \
                                                           / (model_score[:] + 1e-6)
        for i in range(1, c):
            slope_margin_percent[:, indices_slope[:, i]] = (slope_s[:, indices_slope[:, i - 1]]
                                                            - slope_s[:, indices_slope[:, i]] + 1e-6) \
                                                           / (slope_s[:, indices_slope[:, i - 1]] + 1e-6)

        # relu去除负值
        slope_margin_percent = slope_margin_percent.view(bs, c, 1, 1, 1)
        saliency_map = (slope_margin_percent * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)

        # 归一化
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def EFCAM_margin_bs(self, data, action_class=None):  # Erasing Feature CAM
        # 这里的每一个部分都是使用了softmax的，否则top1和top5很高但是increase一直为0，混淆矩阵尚未测试
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.float().to(self.dev)
        model_output = self.model(data)
        # TODO：精简化程序margin部分，支持batchsize大于一，支持多卡

        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)

        model_output = F.softmax(model_output, dim=-1)
        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1).contiguous()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()

        slope = torch.zeros(bs, c).to(self.dev)
        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()
                output = F.softmax(output, dim=-1)

                score = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重
                slope[:, i:i + 1] = model_score - score  # score 越小代表覆盖区域越精准

        slope_s = torch.zeros(bs, c).to(self.dev)
        sorted_slope, indices_slope = torch.sort(slope, dim=-1, descending=True)
        # print(sorted_slope, indices_slope)
        sorted_activations = torch.zeros_like(activations)
        for i in range(sorted_activations.size(0)):
            sorted_activations[i, :, :, :, :] = activations[i, indices_slope[i, :], :, :, :]
        with torch.no_grad():
            # saliency_map_a = activations[:, indices_slope[:, 0], :, :, :].detach()
            norm_saliency_map_a = torch.ones_like(sorted_activations[:, 0:1, :, :, :])
            for i in range(c):
                saliency_map_b = sorted_activations[:, i:i + 1, :, :, :].detach()
                # 归一化
                norm_saliency_map_b = self.data_norm(saliency_map_b)
                norm_saliency_map = 1 - self.data_norm(norm_saliency_map_a * norm_saliency_map_b)
                norm_saliency_map = F.interpolate(norm_saliency_map,
                                                  size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()
                output = F.softmax(output, dim=-1)

                score_s = torch.sum(one_hot * output, dim=1).view(-1, 1).contiguous().detach()
                slope_s[:, i:i + 1] = score_s
                norm_saliency_map_a = self.data_norm(norm_saliency_map_a * norm_saliency_map_b)

        # 计算排序后的边缘值
        slope_margin_percent = torch.zeros_like(slope_s)
        slope_margin_percent[:, 0:1] = (model_score[:] - slope_s[:, 0:1] + 1e-6)  # / (model_score[:] + 1e-6)
        for i in range(1, c):
            slope_margin_percent[:, i:i + 1] = (
                    slope_s[:, i - 1:i] - slope_s[:, i:i + 1] + 1e-6)  # / (slope_s[:, i-1:i] + 1e-6)

        # relu去除负值
        slope_margin_percent = slope_margin_percent.view(bs, c, 1, 1, 1)
        saliency_map = (slope_margin_percent * sorted_activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)

        # 归一化
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def in_abla_softplus(self, data, action_class=None):
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.float().to(self.dev)
        model_output = self.model(data)
        # model_output = F.softmax(model_output, dim=1)
        if not action_class:
            action_class = model_output.argmax(dim=1, keepdim=True)
        # pred_class = model_output.argmax(dim=1, keepdim=True)
        one_hot = torch.zeros(model_output.shape, dtype=torch.float).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1, 1, 1, 1).contiguous()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()

        slope = torch.zeros(bs, c, 1, 1, 1).to(self.dev)

        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                # norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                norm_saliency_map = self.data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                output = self.model(data * norm_saliency_map).detach()
                self.reset_hook_value()
                # output = F.softmax(output, dim=1)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1, 1, 1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重
                slope[:, i:i + 1, :, :, :] = (model_score - score) / model_score
                # score_saliency_map += (score * saliency_map).detach()

            # relu去除负值
        # score_saliency_map = F.relu(score_saliency_map)
        saliency_map = (slope * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        # 归一化
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def UOCAM(self, data, action_class=None):  # union OoD
        # 这里的每一个部分都是使用了softmax的，否则top1和top5很高但是increase一直为0，混淆矩阵尚未测试
        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.to(self.dev)
        model_output = self.model(data)
        # TODO：精简化程序margin部分，支持batchsize大于一，支持多卡

        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)

        model_output = F.softmax(model_output, dim=-1)
        model_output = - torch.log(model_output)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1).contiguous().detach()

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()

        # slope = torch.zeros(bs, c).double().to(self.dev)
        slope = torch.zeros(bs, c).to(self.dev)
        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = activations[:, i:i + 1, :, :, :].detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                saliency_map = self.data_norm(saliency_map)

                ret = saliency_map.mean(dim=(-3, -2, -1), keepdim=False)
                centre_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) <= ret, torch.zeros_like(data), data)
                surround_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) > ret, torch.zeros_like(data), data)
                # mask = torch.where(norm_saliency_map.repeat(1, 3, 1, 1, 1) <= ret, torch.zeros_like(data), torch.ones_like(data))
                del saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                cen_logit = self.model(centre_data).detach()
                self.reset_hook_value()
                cen_logit = F.softmax(cen_logit, dim=1)
                cen_logit = - torch.log(cen_logit)
                cen_logit = torch.sum(one_hot * cen_logit, dim=1).view(-1, 1).contiguous().detach()

                self.model.zero_grad()
                sur_logit = self.model(surround_data).detach()
                self.reset_hook_value()
                sur_logit = F.softmax(sur_logit, dim=1)
                sur_logit = - torch.log(sur_logit)
                sur_logit = torch.sum(one_hot * sur_logit, dim=1).view(-1, 1).contiguous().detach()

                alpha = 0.75
                slope[:, i:i + 1] = (
                        alpha * (model_score - cen_logit) - (1 - alpha) * (model_score - sur_logit)).detach()

        slope = F.softmax(slope, dim=-1)

        # 归一化
        saliency_map = (slope.view(bs, c, 1, 1, 1) * activations).sum(1, keepdim=True)
        # 归一化
        score_saliency_map = self.data_norm(F.relu(saliency_map))
        # score_saliency_map = F.interpolate(score_saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def BICAM_Appendix(self, data, action_class=None, alpha=1, k=100, fix_ret=False):  # union OoD
        # 这里的每一个部分都是使用了softmax的，否则top1和top5很高但是increase一直为0，混淆矩阵尚未测试
        self.model.eval().to(self.dev)
        self.model.zero_grad()
        # action recognition
        data = data.to(self.dev)
        model_output = self.model(data)
        # TODO：精简化程序margin部分，支持batchsize大于一，支持多卡

        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)

        model_output = F.softmax(model_output, dim=-1) + 1e-40
        model_output = - torch.log(model_output)

        one_hot = torch.zeros_like(model_output).to(self.dev)

        one_hot = one_hot.scatter_(1, action_class, 1)

        model_score = torch.sum(one_hot * model_output, dim=1).view(1, 1).contiguous().detach()

        activations = self.activations[-1].detach()

        self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()

        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()
        feed_bs = 32

        # slope = torch.zeros(bs, c).double().to(self.dev)
        slope = torch.zeros(bs, c).to(self.dev)
        with torch.no_grad():
            for i in range(0, c, feed_bs):  # c=256 这段代码是一个 for 循环，它会从 0 开始，每次增加 feed_bs 的步长，直到达到或超过 c（256）为止。

                # 获取第i层特征图,并进行上采样操作
                saliency_map = activations[:, i:i + feed_bs, :, :, :].view(feed_bs, 1, t, v, m).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear',
                                             align_corners=True)
                # 归一化
                saliency_map = self.data_norm(saliency_map)

                # ret = saliency_map.mean(dim=(-3, -2, -1), keepdim=False)
                # centre_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) <= ret, torch.zeros_like(data), data)
                # surround_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) > ret, torch.zeros_like(data), data)

                if fix_ret:
                    ret = 0.5
                else:
                    ret = saliency_map.mean(dim=(-3, -2, -1), keepdim=True)

                saliency_map = saliency_map.repeat(1, 3, 1, 1, 1) - ret

                centre_data = torch.sigmoid(k * saliency_map) * data
                surround_data = torch.sigmoid(-k * saliency_map) * data

                del saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                self.model.zero_grad()
                cen_logit = self.model(centre_data).detach()
                self.reset_hook_value()

                cen_logit = F.softmax(cen_logit, dim=1) + 1e-40

                cen_logit = - torch.log(cen_logit)

                cen_logit = torch.sum(one_hot * cen_logit, dim=1).view(1, -1)
                # print(cen_logit)
                self.model.zero_grad()
                sur_logit = self.model(surround_data).detach()
                self.reset_hook_value()
                sur_logit = F.softmax(sur_logit, dim=1) + 1e-40

                sur_logit = - torch.log(sur_logit)

                sur_logit = torch.sum(one_hot * sur_logit, dim=1).view(1, -1)

                # print(model_score)
                # print(cen_logit)
                slope[:, i:i + feed_bs] = (
                        alpha * (model_score - cen_logit) - (1 - alpha) * (model_score - sur_logit)).detach()
                # print(slope)

        slope = F.softmax(slope, dim=-1)

        # 归一化
        saliency_map = (slope.view(bs, c, 1, 1, 1) * activations).sum(1, keepdim=True)

        # 归一化
        score_saliency_map = self.data_norm(F.relu(saliency_map))
        # score_saliency_map = F.interpolate(score_saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        score_saliency_map = self.reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def cut_skeleton(self, data):
        """
        In order to make the skeleton data to five
        :param data: the skeleton data shape is (N, C, T, V, M)
        :return: five body:left_hand, right_hand, left_leg, right_leg, trunk
        """
        N, C, T, V, M = data.size()

        # 定义各个肢体的矩阵
        left_hand = np.zeros((25, 3))
        left_hand[[8, 9, 10, 11, 23, 24], :] = 1
        left_hands = np.tile(left_hand, (T * M, 1))
        left_hands = torch.from_numpy(left_hands)
        left_hands = self.data_reshape(data, left_hands).to(self.dev)

        right_hand = np.zeros((25, 3))
        right_hand[[4, 5, 6, 7, 21, 22], :] = 1
        right_hands = np.tile(right_hand, (T * M, 1))
        right_hands = torch.from_numpy(right_hands)
        right_hands = self.data_reshape(data, right_hands).to(self.dev)

        left_leg = np.zeros((25, 3))
        left_leg[[16, 17, 18, 19], :] = 1
        left_legs = np.tile(left_leg, (T * M, 1))
        left_legs = torch.from_numpy(left_legs)
        left_legs = self.data_reshape(data, left_legs).to(self.dev)

        right_leg = np.zeros((25, 3))
        right_leg[[12, 13, 14, 15], :] = 1
        right_legs = np.tile(right_leg, (T * M, 1))
        right_legs = torch.from_numpy(right_legs)
        right_legs = self.data_reshape(data, right_legs).to(self.dev)

        trunk = np.zeros((25, 3))
        trunk[[0, 1, 2, 3, 20], :] = 1
        trunks = np.tile(trunk, (T * M, 1))
        trunks = torch.from_numpy(trunks)
        trunks = self.data_reshape(data, trunks).to(self.dev)

        return left_hands, right_hands, left_legs, right_legs, trunks

    def permutation_combination(self, data, label, left_hand, right_hand, left_leg, right_leg, trunk):
        """
        In order to permutation and combination the skeleton of human
        :param label:
        :param data: the translate of data ,shape is (N, T, V, M, C)
        :param left_hand: left hand of human skeleton: 9, 10, 11, 12, 24, 25
        :param right_hand: right hand of human skeleton: 5, 6, 7, 8 ,22, 23
        :param left_leg: left leg of human skeleton: 17, 18, 19, 20
        :param right_leg: right leg of human skeleton: 13, 14, 15, 16
        :param trunk: trunk of human skeleton: 1, 2, 3, 4, 21
        :return:  source_data: 32 permutations and combinations
        """
        source_data = {}
        self.model.eval()
        self.model.zero_grad()
        # <editor-fold desc="single. 单个一组 5 个">
        left_hand = (left_hand * data).float().to(self.dev)
        with torch.no_grad():
            left_hand_out = self.model(left_hand)
            probability_left_hand = (torch.nn.functional.softmax(left_hand_out[0], dim=-1))[label].item()
        source_data["left_hand"] = probability_left_hand

        right_hand = (right_hand * data).float().to(self.dev)
        with torch.no_grad():
            right_hand_out = self.model(left_hand)
            probability_right_hand = (torch.nn.functional.softmax(right_hand_out[0], dim=-1))[label].item()
        source_data["right_hand"] = probability_right_hand

        left_leg = (left_leg * data).float().to(self.dev)
        with torch.no_grad():
            left_leg_out = self.model(left_leg)
            probability_left_leg = (torch.nn.functional.softmax(left_leg_out[0], dim=-1))[label].item()
        source_data["left_leg"] = probability_left_leg

        right_leg = (right_leg * data).float().to(self.dev)
        with torch.no_grad():
            right_leg_out = self.model(right_leg)
            probability_right_leg = (torch.nn.functional.softmax(right_leg_out[0], dim=-1))[label].item()
        source_data["right_leg"] = probability_right_leg

        trunk = (trunk * data).float().to(self.dev)
        with torch.no_grad():
            trunk_out = self.model(right_leg)
            probability_trunk = (torch.nn.functional.softmax(trunk_out[0], dim=-1))[label].item()
        source_data["trunk"] = probability_trunk
        # </editor-fold>

        # <editor-fold desc="two. 两个一组 10个">
        data = data.to(self.dev)
        one_two = left_hand + right_hand
        one_two = (data * one_two).float().to(self.dev)
        with torch.no_grad():
            one_two_out = self.model(one_two)
            probability_one_two = (torch.nn.functional.softmax(one_two_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand"] = probability_one_two

        one_three = left_hand + left_leg
        one_three = (data * one_three).float().to(self.dev)
        with torch.no_grad():
            one_three_out = self.model(one_three)
            probability_one_three = (torch.nn.functional.softmax(one_three_out[0], dim=-1))[label].item()
        source_data["left_hand,left_leg"] = probability_one_three

        one_four = left_hand + right_leg
        one_four = (data * one_four).float().to(self.dev)
        with torch.no_grad():
            one_four_out = self.model(one_four)
            probability_one_four = (torch.nn.functional.softmax(one_four_out[0], dim=-1))[label].item()
        source_data["left_hand,right_leg"] = probability_one_four

        one_five = left_hand + trunk
        one_five = (data * one_five).float().to(self.dev)
        with torch.no_grad():
            one_five_out = self.model(one_five)
            probability_one_five = (torch.nn.functional.softmax(one_five_out[0], dim=-1))[label].item()
        source_data["left_hand,trunk"] = probability_one_five

        two_three = right_hand + left_leg
        two_three = (data * two_three).float().to(self.dev)
        with torch.no_grad():
            two_three_out = self.model(two_three)
            probability_two_three = (torch.nn.functional.softmax(two_three_out[0], dim=-1))[label].item()
        source_data["right_hand,left_leg"] = probability_two_three

        two_four = right_hand + right_leg
        two_four = (data * two_four).float().to(self.dev)
        with torch.no_grad():
            two_four_out = self.model(two_four)
            probability_two_four = (torch.nn.functional.softmax(two_four_out[0], dim=-1))[label].item()
        source_data["right_hand,right_leg"] = probability_two_four

        two_five = right_hand + trunk
        two_five = (data * two_five).float().to(self.dev)
        with torch.no_grad():
            two_five_out = self.model(two_five)
            probability_two_five = (torch.nn.functional.softmax(two_five_out[0], dim=-1))[label].item()
        source_data["right_hand,trunk"] = probability_two_five

        three_four = left_leg + right_leg
        three_four = (data * three_four).float().to(self.dev)
        with torch.no_grad():
            three_four_out = self.model(three_four)
            probability_three_four = (torch.nn.functional.softmax(three_four_out[0], dim=-1))[label].item()
        source_data["left_leg,right_leg"] = probability_three_four

        three_five = left_leg + trunk
        three_five = (data * three_five).float().to(self.dev)
        with torch.no_grad():
            three_five_out = self.model(three_five)
            probability_three_five = (torch.nn.functional.softmax(three_five_out[0], dim=-1))[label].item()
        source_data["left_leg,trunk"] = probability_three_five

        four_five = right_leg + trunk
        four_five = (data * four_five).float().to(self.dev)
        with torch.no_grad():
            four_five_out = self.model(four_five)
            probability_four_five = (torch.nn.functional.softmax(four_five_out[0], dim=-1))[label].item()
        source_data["right_leg,trunk"] = probability_four_five
        # </editor-fold>

        # <editor-fold desc="three. 三个一组 10个">
        one_two_three = one_two + left_leg
        one_two_three = (data * one_two_three).float().to(self.dev)
        with torch.no_grad():
            one_two_three_out = self.model(one_two_three)
            probability_one_two_three = (torch.nn.functional.softmax(one_two_three_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,left_leg"] = probability_one_two_three

        one_two_four = one_two + right_leg
        one_two_four = (data * one_two_four).float().to(self.dev)
        with torch.no_grad():
            one_two_four_out = self.model(one_two_four)
            probability_one_two_four = (torch.nn.functional.softmax(one_two_four_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,right_leg"] = probability_one_two_four

        one_two_five = one_two + trunk
        one_two_five = (data * one_two_five).float().to(self.dev)
        with torch.no_grad():
            one_two_five_out = self.model(one_two_five)
            probability_one_two_five = (torch.nn.functional.softmax(one_two_five_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,trunk"] = probability_one_two_five

        one_three_four = one_three + right_leg
        one_three_four = (data * one_three_four).float().to(self.dev)
        with torch.no_grad():
            one_three_four_out = self.model(one_three_four)
            probability_one_three_four = (torch.nn.functional.softmax(one_three_four_out[0], dim=-1))[label].item()
        source_data["left_hand,left_leg,right_leg"] = probability_one_three_four

        one_three_five = one_three + trunk
        one_three_five = (data * one_three_five).float().to(self.dev)
        with torch.no_grad():
            one_three_five_out = self.model(one_three_five)
            probability_one_three_five = (torch.nn.functional.softmax(one_three_five_out[0], dim=-1))[label].item()
        source_data["left_hand,left_leg,trunk"] = probability_one_three_five

        one_four_five = one_four + trunk
        one_four_five = (data * one_four_five).float().to(self.dev)
        with torch.no_grad():
            one_four_five_out = self.model(one_four_five)
            probability_one_four_five = (torch.nn.functional.softmax(one_four_five_out[0], dim=-1))[label].item()
        source_data["left_hand,right_leg,trunk"] = probability_one_four_five

        two_three_four = two_three + right_leg
        two_three_four = (data * two_three_four).float().to(self.dev)
        with torch.no_grad():
            two_three_four_out = self.model(two_three_four)
            probability_two_three_four = (torch.nn.functional.softmax(two_three_four_out[0], dim=-1))[label].item()
        source_data["right_hand,left_leg,right_leg"] = probability_two_three_four

        two_three_five = two_three + trunk
        two_three_five = (data * two_three_five).float().to(self.dev)
        with torch.no_grad():
            two_three_five_out = self.model(two_three_five)
            probability_two_three_five = (torch.nn.functional.softmax(two_three_five_out[0], dim=-1))[label].item()
        source_data["right_hand,left_leg,trunk"] = probability_two_three_five

        two_four_five = two_four + trunk
        two_four_five = (data * two_four_five).float().to(self.dev)
        with torch.no_grad():
            two_four_five_out = self.model(two_four_five)
            probability_two_four_five = (torch.nn.functional.softmax(two_four_five_out[0], dim=-1))[label].item()
        source_data["right_hand,right_leg,trunk"] = probability_two_four_five

        three_four_five = three_four + trunk
        three_four_five = (data * three_four_five).float().to(self.dev)
        with torch.no_grad():
            three_four_five_out = self.model(three_four_five)
            probability_three_four_five = (torch.nn.functional.softmax(three_four_five_out[0], dim=-1))[label].item()
        source_data["left_leg,right_leg,trunk"] = probability_three_four_five
        # </editor-fold>

        # <editor-fold desc="four. 四个一组 5个">
        one_two_three_four = one_two_three + right_leg
        one_two_three_four = (data * one_two_three_four).float().to(self.dev)
        with torch.no_grad():
            one_two_three_four_out = self.model(one_two_three_four)
            probability_one_two_three_four = (torch.nn.functional.softmax(one_two_three_four_out[0], dim=-1))[
                label].item()
        source_data["left_hand,right_hand,left_leg,right_leg"] = probability_one_two_three_four

        one_two_three_five = one_two_three + trunk
        one_two_three_five = (data * one_two_three_five).float().to(self.dev)
        with torch.no_grad():
            one_two_three_five_out = self.model(one_two_three_five)
            probability_one_two_three_five = (torch.nn.functional.softmax(one_two_three_five_out[0], dim=-1))[
                label].item()
        source_data["left_hand,right_hand,left_leg,trunk"] = probability_one_two_three_five

        one_two_four_five = one_two_four + trunk
        one_two_four_five = (data * one_two_four_five).float().to(self.dev)
        with torch.no_grad():
            one_two_four_five_out = self.model(one_two_four_five)
            probability_one_two_four_five = (torch.nn.functional.softmax(one_two_four_five_out[0], dim=-1))[
                label].item()
        source_data["left_hand,right_hand,right_leg,trunk"] = probability_one_two_four_five

        one_three_four_five = one_three_four + trunk
        one_three_four_five = (data * one_three_four_five).float().to(self.dev)
        with torch.no_grad():
            one_three_four_five_out = self.model(one_three_four_five)
            probability_one_three_four_five = (torch.nn.functional.softmax(one_three_four_five_out[0], dim=-1))[
                label].item()
        source_data["left_hand,left_leg,right_leg,trunk"] = probability_one_three_four_five

        two_three_four_five = two_three_four + trunk
        two_three_four_five = (data * two_three_four_five).float().to(self.dev)
        with torch.no_grad():
            two_three_four_five_out = self.model(two_three_four_five)
            probability_two_three_four_five = (torch.nn.functional.softmax(two_three_four_five_out[0], dim=-1))[
                label].item()
        source_data["right_hand,left_leg,right_leg,trunk"] = probability_two_three_four_five
        # </editor-fold>

        # <editor-fold desc="five. 5个一组 1个">
        completion = data.float().to(self.dev)
        with torch.no_grad():
            completion_out = self.model(completion)
            label_new = completion_out.argmax(dim=1, keepdim=True)
            probability_completion = (torch.nn.functional.softmax(completion_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,left_leg,right_leg,trunk"] = probability_completion
        # </editor-fold>

        return source_data

    def compute_shapley_values(self, unique_keys, source_data):
        """
        In order to Compute shapley values of each permutation
        :param unique_keys: the permutation of human skeleton
        :param source_data: the probability of each permutation
        :return: the shapley values of each permutation human skeleton
        """
        # 所有的联盟
        all_coalitions = [list(j) for i in range(len(unique_keys)) for j in
                          itertools.combinations(unique_keys, r=i + 1)]

        # 联盟成员数
        n = len(unique_keys)
        shapley_value_list = []

        # 遍历每一个成员
        for k in unique_keys:

            # 包含k的联盟
            k_coalition = []
            # 剔除k后的联盟
            no_k_coalition = []
            # 夏普利值
            k_shapley_value = []

            # 找出所有包含k的联盟，并构造剔除k后的联盟
            for c in copy.deepcopy(all_coalitions):
                if k in c:
                    # 联盟贡献
                    k_coalition.append((copy.deepcopy(c), source_data[','.join(c)]))
                    # 剔除k后的贡献
                    c.remove(k)
                    if len(c) == 0:
                        no_k_coalition.append((c, 0))
                    else:
                        no_k_coalition.append((c, source_data[','.join(c)]))

            # file_txt.write("=================================================================" + '\n')
            # file_txt.write(str(k) + "的联盟" + str(k_coalition) + '\n')
            # file_txt.write("=================================================================" + '\n')
            # file_txt.write("剔除" + str(k) + "的联盟" + str(no_k_coalition) + '\n')
            # print('k的联盟:', k_coalition)
            # print('剔除k的联盟:', no_k_coalition)

            # 遍历包含k的联盟，并计算夏普利值
            shapley_value = 0
            for i in range(len(k_coalition)):
                s = len(k_coalition[i][0])
                # 成员k的边际贡献
                k_payoff = k_coalition[i][1] - no_k_coalition[i][1]
                # 联盟的权重系数
                k_weight = math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
                shapley_value += k_payoff * k_weight

            k_shapley_value.append((k, shapley_value))
            shapley_value_list.append(k_shapley_value)

            # 联盟的权重
            # file_txt.write("=================================================================" + '\n')
            # file_txt.write(str(k) + "夏普利值:" + str(k_shapley_value) + '\n')
            # print('夏普利值：', k_shapley_value)

        print('各个部位的夏普利值：', shapley_value_list)
        # file_txt.write("=================================================================" + '\n')
        # file_txt.write("各个部位的夏普利值:" + str(shapley_value_list) + '\n')

        return shapley_value_list

    def saliency_map(self, shapley_value, data):
        """
        In order to make the shapley value visualization and computer the evaluate
        :param shapley_value: each human skeleton shapley value
        :param data: the human skeleton data to get the each body of human
        :return: the evaluate and visualization of shapley value
        """
        left_hand, right_hand, left_leg, right_leg, trunk = self.cut_skeleton(data)
        shapley_value = np.array(shapley_value).squeeze(1)

        saliency_map = left_hand * float(shapley_value[0][1]) + right_hand * float(shapley_value[1][1]) + \
                       left_leg * float(shapley_value[2][1]) + right_leg * float(shapley_value[3][1]) + \
                       trunk * float(shapley_value[4][1])
        # base_saliency_map = self.base_cam(data)
        # base_saliency_map = base_saliency_map.cpu().to(self.dev)
        # base_saliency_map = base_saliency_map.to(self.dev)
        base_saliency_map, _, _ = self.GradCampp(data)
        base_saliency_map = base_saliency_map
        saliency_map = torch.sum(saliency_map, dim=1, keepdim=True).float()
        saliency_map = saliency_map * saliency_map
        # print(saliency_map)
        # print(base_saliency_map)
        saliency_map = saliency_map * base_saliency_map

        return saliency_map

    def ShapleyCam(self, data, label):
        """
        In order to get the saliency_map of
        :param label:
        :param data: the original human skeleton data, shape is (N, C, T, V, M)
        :return:
        """
        # print("11111111111111111")
        # print(label)
        left_hand, right_hand, left_leg, right_leg, trunk = self.cut_skeleton(data)
        source_data = self.permutation_combination(data, label, left_hand, right_hand, left_leg, right_leg, trunk)
        shapley_value = self.compute_shapley_values(unique_keys, source_data)
        saliency_map = self.saliency_map(shapley_value, data)

        return saliency_map

    def base_cam(self, data, action_class=None):
        """

        """
        # self.model=self.model.double()
        self.model.eval()
        self.model.zero_grad()

        # action recognition
        data = data.to(self.dev)
        # data = data.double().to(self.dev)
        model_output = self.model(data)

        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)

        loss = torch.sum(one_hot * model_output)
        loss.backward(retain_graph=True)

        # 获取梯度
        gradients = self.gradients[0].detach()
        activations = self.activations[-1].detach()
        self.reset_hook_value()

        bs, c, t, v = gradients.shape  # 实际处理中bs和m已经合并
        gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = gradients.shape

        # 核心处理方法
        alpha = gradients.view(bs, c, -1).mean(2)
        weights = alpha.view(bs, c, 1, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = torch.nn.functional.relu(saliency_map).detach()

        saliency_map = self.reshape_saliency_map(saliency_map, data)

        # alpha_num = gradients.pow(2)
        # alpha_denom = 2 * alpha_num + \
        #               activations.mul(gradients.pow(3)).sum(dim=(2, 3, 4), keepdim=True)
        # alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        # alpha = alpha_num.div(alpha_denom + 1e-6)

        # relu_grad = torch.nn.functional.relu((model_output * one_hot).sum(dim=1, keepdim=False).exp() * gradients)
        # weights = (alpha * relu_grad).view(bs, c, -1).sum(-1).view(bs, c, 1, 1, 1)

        # saliency_map = (weights * activations).sum(1, keepdim=True)
        # saliency_map = torch.nn.functional.relu(saliency_map).detach()
        # saliency_map = self.reshape_saliency_map(saliency_map, data)
        # print(saliency_map)
        return saliency_map

    def data_reshape(self, data, mask):
        N, C, T, V, M = data.size()
        reshaped_data = torch.reshape(mask, [N, T, V, M, C])
        reshaped_data = reshaped_data.permute(0, 4, 1, 2, 3).contiguous()

        return reshaped_data

    def IMCam(self, data, action_class=None):

        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.to(self.dev)
        model_output = self.model(data)
        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)
        target_one_hot = torch.zeros_like(model_output).to(self.dev)
        target_one_hot = target_one_hot.scatter_(1, action_class, 1)

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.size()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.size()

        slope = torch.zeros(bs, c).to(self.dev)

        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map,
                                             size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                ret = norm_saliency_map.mean(dim=(-3, -2, -1), keepdim=False)
                drop_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1, 1) < (ret),
                                        torch.zeros_like(data), data)
                output = self.model(drop_data).detach()
                self.reset_hook_value()

                # output = F.softmax(output, dim=1)
                predict_class = output.argmax(dim=1, keepdim=True)
                pred_one_hot = torch.zeros_like(output).to(output.device)
                pred_one_hot = pred_one_hot.scatter_(1, predict_class, 1)

                pred_score = torch.sum(pred_one_hot * output, dim=-1, keepdim=False)
                target_score = torch.sum(target_one_hot * output, dim=1)

                slope[:, i:i + 1] = torch.exp(target_score - pred_score + 1)

        saliency_map = (slope.view(bs, c, 1, 1, 1) * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        score_saliency_map = self.data_norm(F.relu(score_saliency_map))
        # score_saliency_map 为所求
        score_saliency_map = F.interpolate(score_saliency_map,
                                           size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def ISGCam(self, data, action_class=None):

        self.model.eval()
        self.model.zero_grad()
        # action recognition
        data = data.to(self.dev)
        model_output = self.model(data)
        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)
        target_one_hot = torch.zeros_like(model_output).to(self.dev)
        target_one_hot = target_one_hot.scatter_(1, action_class, 1)

        activations = self.activations[-1].detach()
        self.reset_hook_value()
        bs, c, t, v = activations.size()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.size()

        slope1_target = torch.zeros(bs, c).to(self.dev)
        slope1_contrast = torch.zeros(bs, c).to(self.dev)

        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map,
                                             size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                # 归一化
                norm_saliency_map = self.data_norm(saliency_map)
                ret = norm_saliency_map.mean(dim=(-3, -2, -1), keepdim=False)
                drop_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1, 1) < (ret),
                                        torch.zeros_like(data), data)
                output = self.model(drop_data).detach()
                self.reset_hook_value()

                # output = F.softmax(output, dim=1)
                predict_class = output.argmax(dim=1, keepdim=True)
                pred_one_hot = torch.zeros_like(output).to(output.device)
                pred_one_hot = pred_one_hot.scatter_(1, predict_class, 1)

                pred_score = torch.sum(pred_one_hot * output, dim=-1, keepdim=False)
                target_score = torch.sum(target_one_hot * output, dim=1)

                slope1_target[:, i:i + 1] = target_score
                slope1_contrast[:, i:i + 1] = torch.exp(target_score - pred_score + 1)

            sorted_slope, indices_slope = torch.sort(slope1_target, dim=-1, descending=True)
            sorted_contrast_slope = torch.zeros_like(slope1_contrast)
            sorted_activations = torch.zeros_like(activations)

            for i in range(sorted_activations.size(0)):
                sorted_activations[i, :, :, :, :] = activations[i, indices_slope[i, :], :, :, :]
                sorted_contrast_slope[i, :] = slope1_contrast[i, indices_slope[i, :]]
            accum_activation = sorted_activations[:, 0:1, :, :, :]
            accm_slope = torch.zeros_like(sorted_contrast_slope)

            for i in range(1, c):
                accum_activation = accum_activation + sorted_activations[:, i:i + 1, :, :, :]
                norm_accum_activation = F.interpolate(accum_activation, size=(data.size(2), v, m), mode='trilinear',
                                                      align_corners=True)
                # norm
                norm_accum_activation = self.data_norm(norm_accum_activation)
                ret = norm_accum_activation.mean(dim=(-3, -2, -1), keepdim=False)
                drop_data = torch.where(norm_accum_activation.repeat(1, 3, 1, 1, 1) < (ret),
                                        torch.zeros_like(data), data)
                sub_logit = self.model(drop_data).detach()
                self.reset_hook_value()

                predict_class = sub_logit.argmax(dim=1, keepdim=True)
                pred_one_hot = torch.zeros_like(sub_logit).to(sub_logit.device)
                pred_one_hot = pred_one_hot.scatter_(1, predict_class, 1)
                target_score = torch.sum(target_one_hot * sub_logit, dim=-1, keepdim=False)
                pred_score = torch.sum(pred_one_hot * sub_logit, dim=-1, keepdim=False)
                accm_slope[:, i:i + 1] = sorted_contrast_slope[:, i:i + 1] / torch.exp(target_score - pred_score + 1)

        saliency_map = (accm_slope.view(bs, c, 1, 1, 1) * sorted_activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        score_saliency_map = self.data_norm(score_saliency_map)
        # score_saliency_map 为所求
        score_saliency_map = F.interpolate(score_saliency_map,
                                           size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        return score_saliency_map.detach(), model_output.detach(), action_class.detach()

    def channel_norm(self, data):
        bs, c, t, v, m = data.shape
        data_min = data.view(bs, c, -1).min(dim=-1, keepdim=True).values.view(bs, c, 1, 1, 1)
        data_max = data.view(bs, c, -1).max(dim=-1, keepdim=True).values.view(bs, c, 1, 1, 1)
        # denominator = torch.where(
        #     (data_max-data_min) != 0., data_max-data_min, torch.tensor(1.).double().to(self.dev))
        denominator = torch.where(
            (data_max - data_min) != 0., data_max - data_min, torch.ones_like(data_max).to(self.dev))
        return (data - data_min) / denominator

    def data_norm(self, data):
        bs, c, t, v, m = data.shape
        data_min = data.view(bs, -1).min(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
        data_max = data.view(bs, -1).max(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
        # denominator = torch.where(
        #     (data_max-data_min) != 0., data_max-data_min, torch.tensor(1.).double().to(self.dev))
        denominator = torch.where(
            (data_max - data_min) != 0., data_max - data_min, torch.ones_like(data_max).to(self.dev))

        return (data - data_min) / denominator

    def reshape_saliency_map(self, saliency_map, data):
        bs, c, t, v, m = saliency_map.shape
        saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        saliency_map = self.data_norm(saliency_map)
        return saliency_map

    def evaluate(self, data, saliency_map, label=None):
        '''评价指标'''
        if saliency_map.dtype == torch.double:
            self.model = self.model.double()
            data = data.double()
        self.model.eval()
        self.model.zero_grad()

        # action recognition

        ''' --------- 事实证明 Drop方法不能用 -------- 
            因为会直接改变节点位置
            又无法移除，因为是时空图
        '''
        # drop_data = data + (1 - saliency_map)

        # 下面这种方法对结果影响的随机性有点大
        '''
        print(data.shape, saliency_map.shape)
        theta = torch.rand(saliency_map.shape) * 2 * np.pi 
        fai = torch.rand(saliency_map.shape) * 2 * np.pi
        disturb_mat = torch.zeros(data.shape)
        disturb_mat[:,0,:,:,:] = (1 - saliency_map) * torch.sin(theta) * torch.cos(fai)
        disturb_mat[:,1,:,:,:] = (1 - saliency_map) * torch.sin(theta) * torch.sin(fai)
        disturb_mat[:,2,:,:,:] = (1 - saliency_map) * torch.cos(theta)
        drop_data = data + abs(disturb_mat * 0.1)'''

        # 试一下直接置零，似乎能用

        ## 统计非零元素
        bs, c, t, v, m = saliency_map.shape
        data = data.to(self.dev)

        '''
        saliency_map = F.relu(saliency_map)
        if self.arg.data_level == 'test_set':

            # sort nodes in saliency map
            indices = torch.ones((bs, c, t, v, m), dtype=torch.int) * 15000
            for i in range(bs):
                mask = self.mask[i]
                saliency_map_cache = (saliency_map[i][:, 0:mask[0], :, 0:mask[1]]).view(1, 1, 1, -1)
                _, indices_ = torch.sort(saliency_map_cache, dim=-1, descending=True)
                indices[i][:, 0:mask[0], :, 0:mask[1]] = indices_.view(c, mask[0], v, mask[1])
            nonzero_num = self.mask[:, 0] * self.mask[:, 1] * (c * v) # c = 1 for saliency map

            # convenient but unstable algorithm, since sort algorithm in torch is unstable
            # saliency_map[self.mask == 0] = -1
            # saliency_map = saliency_map.view(bs, -1)
            # _, indices = torch.sort(saliency_map, dim=-1, descending=True)
            # indices = indices2.view(bs, c, t, v, m)
            # saliency_map = saliency_map.view(bs, c, t, v, m)
            # saliency_map[saliency_map == -1] = 0
            # indices[self.mask == 0] = 15000
        else:
            nonzero_num = [c * t * v * m]

            # sort saliency map
            _, indices = torch.sort(saliency_map.view(bs, -1), dim=-1, descending=True)
            indices = indices.view(bs, c, t, v, m)

        '''

        if self.arg.data_level == 'test_set':
            mask = self.mask[0]
            cache = saliency_map[:, :, :mask[0], :, :mask[1]].cpu()
        else:
            cache = saliency_map.cpu()

        ## 1. 计算ave drop和ave increase
        threshold = np.percentile(cache, 50)
        drop_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) > threshold,
                                data, torch.zeros_like(data).to(self.dev))

        with torch.no_grad():
            self.model.zero_grad()
            model_output = self.model(data)
            model_output = F.softmax(model_output, dim=-1)
            self.reset_hook_value()

            self.model.zero_grad()
            drop_output = self.model(drop_data)
            drop_output = F.softmax(drop_output, dim=-1)
            self.reset_hook_value()

        one_hot = torch.zeros(model_output.shape, dtype=torch.float32).to(self.dev)
        model_class = model_output.argmax(dim=1, keepdim=True)
        one_hot = one_hot.scatter_(1, model_class, 1)

        score = torch.sum(one_hot * model_output, dim=1)
        drop_score = torch.sum(one_hot * drop_output, dim=1)

        average_drop = (F.relu(score - drop_score) / score).sum().detach().cpu().numpy()
        increase = (score < drop_score).sum().detach().cpu().numpy()

        ## 2. 计算insertion 和 deletion
        drop_num = 100 // drop_stride
        threshold = [np.percentile(cache, i * drop_stride) for i in range(drop_num, 0, -1)]
        saliency_map = saliency_map.repeat(1, 3, 1, 1, 1)
        # print(threshold)

        # insersion_list = torch.zeros((1, drop_num)).double().to(self.dev)
        insersion_list = []

        for drop_radio in range(drop_num):
            drop_data = torch.where(saliency_map > threshold[drop_radio], data,
                                    torch.zeros_like(data)).detach()

            self.model.zero_grad()
            with torch.no_grad():
                drop_logit = self.model(drop_data)
                self.reset_hook_value()
                drop_logit = F.softmax(drop_logit, dim=-1)
                drop_score = torch.sum(one_hot * drop_logit, dim=1)
            insersion_list.append(drop_score[0].detach().cpu().numpy())
            # insersion_list[:, drop_radio] = drop_score[0].detach()

        insersion_list = np.array(insersion_list)

        # insersion_list = insersion_list.cpu().numpy()
        # print(insersion_list)
        insersion_auc = insersion_list.sum() * drop_stride / 100
        # insersion_auc = insersion_auc.cpu().numpy()

        deletion_list = []

        for drop_radio in range(drop_num):
            drop_data = torch.where(saliency_map > threshold[drop_radio],
                                    torch.zeros_like(data), data).detach()

            self.model.zero_grad()
            with torch.no_grad():
                drop_logit = self.model(drop_data)
                self.reset_hook_value()
                drop_logit = F.softmax(drop_logit, dim=-1)
                drop_score = torch.sum(one_hot * drop_logit, dim=1)
            deletion_list.append(drop_score[0].detach().cpu().numpy())

        deletion_list = np.array(deletion_list)
        deletion_auc = deletion_list.sum() * drop_stride / 100
        # deletion_auc = deletion_auc.cpu().numpy()
        file_txt.write("Insertion Curve:" + '\n' + str(insersion_list) + '\n')
        file_txt.write("Insertion AUC:" + str(insersion_auc) + '\n')
        file_txt.write("Deletion Curve:" + '\n' + str(deletion_list) + '\n')
        file_txt.write("Deletion AUC:" + str(deletion_auc) + '\n')
        if self.arg.data_level != 'test_set':
            print("Insertion Curve:\n", insersion_list)
            print("\nInsertion AUC:", insersion_auc)
            print("\nDeletion Curve:\n", deletion_list)
            print("\nDeletion AUC:", deletion_auc)
        print(average_drop, increase, insersion_auc, deletion_auc)
        return average_drop, increase, insersion_auc, deletion_auc

    # 渲染骨骼序列和显著性图
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
                            default='S001C001P001R001A007',  # S018C001P058R002A53
                            help='Path to video')
        # NTU-RGB+D 60: S001C001P001R001A007 throw S001C001P001R001A044 headache S001C001P001R001A055 hugging
        # NTU-RGB+D 120: S022C003P061R001A061 接电话 S029C002P080R002A100 后踢腿 S022C003P061R001A112 击掌
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--plot_action',
                            default=True,
                            help='save action as image',
                            type=bool)
        parser.add_argument('--output_dir',
                            default='./work3result_v/',
                            help='Path to save results')
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--run_device',
                            default='cuda:2',
                            help='Dev to use, e.g., "cuda:0", "cuda:1", "cuda:2" or "cpu"')
        parser.add_argument('--cam_type',
                            default='ablation',
                            help='One of gradcam, gradcampp, smoothcam, \
                                ablation, scorecam, ada-gradcam, axiom, l2-caf,shapleycam,bicam,isgcam')
        parser.add_argument('--data_level',
                            default='test_set',
                            help='instance or test_set')
        parser.add_argument('--valid',
                            default='xsub',
                            help='One of xsub and xview, csub, csetup')
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

        parser.add_argument('--alpha', type=float, default=1.0, help="alpha for bicam")
        parser.add_argument('--m', type=int, default=100, help="m for bicam")
        parser.add_argument('--fix_ret', action='store_true', default=False, help="fix ret for bicam")

        args = parser.parse_known_args(namespace=parent_parser)
        # print("mark", parent_parser.valid)
        # parser.set_defaults(
        # config='./config/st_gcn/ntu-{}/demo_skeleton_cam.yaml'.format(parent_parser.valid))

        # NTU60 和 NTU120 数据集自动路径判断
        ntu60_valids = ['xsub', 'xview']
        valid = args[0].valid
        if valid in ntu60_valids:
            config_path = './config/st_gcn/ntu-{}/test.yaml'.format(parent_parser.valid)  # cff,gnnexplainer
        else:
            config_path = './config/st_gcn/ntu120-{}/test.yaml'.format(parent_parser.valid)
        parser.set_defaults(config=config_path)
        parser.set_defaults(print_log=False)
        print(f"Using config: {config_path}")
        # endregion yapf: enable
        return parser
