import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import time
import os
import numpy as np
from numpy import *
from .processor import Processor
from math import sqrt
from torch.nn.functional import cross_entropy
# from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new, plucker
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new
import tools.utils as utils
from torch_geometric.nn import MessagePassing
from tools.utils.visualization_skeleton import plot_action
import random

import csv
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.integrate import simps
# import torchsnooper
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.container")

# 记录结果
# time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
# file_txt = open('result20230306/' + time_now + 'gnnexplianer.txt', mode='a', encoding="utf-8")

max_body = 2
num_joint = 25
max_frame = 300

coeffs = {
    'mask_size': 0.005,
    'mask_ent': 1.0,
}


def data_reshape(data, mask):
    N, C, T, V, M = data.size()
    reshaped_mask = torch.reshape(mask, [N, T, V, M, C])
    reshaped_mask = reshaped_mask.permute(0, 4, 1, 2, 3).contiguous()

    return reshaped_mask


def data_discrete(data):
    N, C, T, V, M = data.size()
    data_mean = torch.zeros_like(data)
    if M == 2:
        mean_x1 = data[:, 0, :, :, 0].mean()
        mean_x2 = data[:, 0, :, :, 1].mean()
        mean_y1 = data[:, 1, :, :, 0].mean()
        mean_y2 = data[:, 1, :, :, 1].mean()
        mean_acc1 = data[:, 2, :, :, 0].mean()
        mean_acc2 = data[:, 2, :, :, 1].mean()

        data_mean[:, 0, :, :, 0] = mean_x1
        data_mean[:, 0, :, :, 1] = mean_x2
        data_mean[:, 1, :, :, 0] = mean_y1
        data_mean[:, 1, :, :, 1] = mean_y2
        data_mean[:, 2, :, :, 0] = mean_acc1
        data_mean[:, 2, :, :, 1] = mean_acc2
    else:
        mean_x = data[:, 0, :, :, :].mean()
        mean_y = data[:, 1, :, :, :].mean()
        mean_acc = data[:, 2, :, :, :].mean()

        data_mean[:, 0, :, :, :] = mean_x
        data_mean[:, 1, :, :, :] = mean_y
        data_mean[:, 2, :, :, :] = mean_acc

    discrete_data = data_mean - data
    discrete_data[discrete_data < 0] = 0
    discrete_data[discrete_data > 0] = 1

    return discrete_data


class Explainer(Processor):
    def __init__(self, argv=None):
        super().__init__(argv)
        # self.epochs = 100
        self.epochs = 50

    def start(self):
        self.dev = self.arg.run_device
        print(self.arg.run_device)
        print("Using {} weights.".format(self.arg.valid))
        explainer_dict = {'gnnexplainer': self.explainer,  # GNNExplainer
                          'cff': self.cff_GNNExplainer,  # CFF
                          }
        self.method = self.arg.exp_type.lower()
        out = self.process_exps(explainer_dict[self.method])

        return

    # @torchsnooper.snoop()
    def process_exps(self, exp_func):
        """
        该模块主要是设置模型解释的样本，分为单实例和多个实例。
        instance：单实例 定性解决问题时使用。
        test_set：多实例 定量分析效果。
        :param :exp_func 所选取的可解释方法。
        """
        # 记录结果
        time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
        os.makedirs('result-forgnne-3-20240306', exist_ok=True)
        file_txt = self.file_txt = open(
            f'result-forgnne-3-20240306/{self.arg.exp_type}-ntu-{self.arg.valid}-{time_now}.txt', mode='a',
            encoding="utf-8")

        if self.arg.data_level == 'instance':
            label_name_path = './resource/ntu_skeleton/label_name.txt'
            with open(label_name_path) as f:
                label_name = f.readlines()
                label_name = [line.rstrip() for line in label_name]
                self.label_name = label_name
            skeleton_name = self.arg.skeleton
            print("Processing skeleton:", skeleton_name)
            file_txt.write("Processing skeleton:" + skeleton_name)
            # skeleton_file = '/data/home/yr/YRR/HDGCN/data/nturgbd_raw/ntuall/'  # ntu120all
            # skeleton_file = './data/NTU-RGB-D/nturgb+d_skeletons/'
            skeleton_file = './data/NTU-RGB-D/ntuall/'
            skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'

            output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_name)
            if not os.path.exists('{}/cam/'.format(output_result_dir)):
                os.makedirs('{}/cam/'.format(output_result_dir))

            data_numpy = read_xyz(
                skeleton_file, max_body=max_body, num_joint=num_joint)
            # data_numpy = plucker(
            #     skeleton_file, max_body=max_body, num_joint=num_joint)
            self.model.eval()
            data = torch.from_numpy(data_numpy)
            data = data.unsqueeze(0)
            data_model = data.float().to(self.dev)
            self.model = self.model.float()
            self.model = self.model.to(self.dev)
            with torch.no_grad():
                # out, _ = self.model(data_model)
                out = self.model(data_model)
                label = out.argmax(dim=1, keepdim=True).item()
                probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
            N, C, T, V, M = data.size()
            x = data.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(N * T * V * M, C)
            node_feature_mask = exp_func(data, x, label)
            # all_fidelity_acc,  all_infidelity_acc, \
            #         all_fidelity_prob,  all_infidelity_prob,Sparsity = \
            #             self.evaluate(node_feature_mask, data,  label, probability)

            if self.arg.plot_action:
                visoutput_result_dir = f'./instanceresult-forcam-20250908/{self.method}-ntu-{self.arg.valid}-{time_now}/'
                output_result_dir = '{}/{}'.format(visoutput_result_dir, skeleton_name)
                if not os.path.exists('{}'.format(output_result_dir)):
                    os.makedirs('{}'.format(output_result_dir))
                C, T, V, M = data_numpy.shape
                node_mask = data_reshape(data, node_feature_mask)
                wise = torch.sum(node_mask, dim=1)
                # wise = wise.unsqueeze(0)
                utils.visualization_skeleton.plot_action(data_numpy, self.model.graph.edge, wise.cpu().numpy(),
                                                         save_dir=output_result_dir,
                                                         save_type=self.arg.exp_type)

            return
        if self.arg.data_level == 'test_set':

            count = 0
            all_fidelity_acc_out = 0

            all_infidelity_acc_out = 0

            all_fidelity_prob_out = 0

            all_infidelity_prob_out = 0
            Sparsity_out = 0
            # 为标准差存储各样本值
            acc_list = []
            prob_list = []
            sparsity_list = []
            # skeleton_file_path = '/data/home/yr/YRR/gcne/data/NTU-RGB-D/nturgb+d_skeletons/nturgb+d_skeletons/'#ntu60
            # skeleton_file_path = '/data/home/yr/YRR/HDGCN/data/nturgbd_raw/ntuall/'#ntu120
            # for root, dirs, files in os.walk(skeleton_file_path):
            #     for file in files:
            #         print("=================================================================")
            #         file_txt.write("=================================================================" + '\n')
            #         file_txt.write(str(datetime.datetime.now()) + '\n')
            #         skeleton_file = skeleton_file_path + file
            #         data_numpy = read_xyz(
            #             skeleton_file, max_body=max_body, num_joint=num_joint)
            #         # data_numpy = plucker(
            #         #     skeleton_file, max_body=max_body, num_joint=num_joint)
            #         self.model.eval()
            #         data = torch.from_numpy(data_numpy)
            #         data = data.unsqueeze(0)
            #         data_model = data.float().to(self.dev)
            #         with torch.no_grad():
            #             out = self.model(data_model)
            #             label = out.argmax(dim=1, keepdim=True).item()
            #             probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
            #         N, C, T, V, M = data.size()
            #         x = data.permute(0, 2, 3, 4, 1).contiguous()
            #         x = x.view(N * T * V * M, C)

            # # output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_file)
            # # if not os.path.exists('{}'.format(output_result_dir)):
            # #     os.makedirs('{}'.format(output_result_dir))
            # visoutput_result_dir = '/data/home/yr/YRR/gcne/result231228/gnnexplainervisadmmgeo/gnnexplainervisadmmgeo'
            # # output_result_dir = '{}/{}'.format(visoutput_result_dir, skeleton_file)
            # output_result_dir = '{}-{}/{}'.format(visoutput_result_dir,time_now, file)
            # if not os.path.exists('{}'.format(output_result_dir)):
            #     os.makedirs('{}'.format(output_result_dir))
            loader = self.data_loader['test']
            for data, label in loader:
                self.model.eval()
                self.model = self.model.to(self.dev)
                # data = torch.from_numpy(data_numpy)
                # data = data.unsqueeze(0)
                data_model = data.float().to(self.dev)
                with torch.no_grad():
                    # out, _ = self.model(data_model)
                    out = self.model(data_model)
                    label = out.argmax(dim=1, keepdim=True).item()
                    probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()

                N, C, T, V, M = data.size()
                x = data.permute(0, 2, 3, 4, 1).contiguous()
                x = x.view(N * T * V * M, C)

                node_feature_mask = exp_func(data, x, label)
                # if self.arg.plot_action:
                #    node_mask = data_reshape(data, node_feature_mask)
                #    wise = torch.sum(node_mask, dim=1)
                #    utils.visualization_skeleton.plot_action(data_numpy, self.model.graph.edge, wise.numpy(),
                #                                             save_dir=output_result_dir,
                #                                             save_type=self.arg.exp_type)
                all_fidelity_acc, all_infidelity_acc, \
                    all_fidelity_prob, all_infidelity_prob, Sparsity = \
                    self.evaluate(node_feature_mask, data, label, probability)

                all_fidelity_acc_out += all_fidelity_acc

                all_infidelity_acc_out += all_infidelity_acc

                all_fidelity_prob_out += all_fidelity_prob

                all_infidelity_prob_out += all_infidelity_prob
                Sparsity_out += Sparsity

                # 用于标准差计算的样本列表
                acc_list.append(all_fidelity_acc)
                prob_list.append(all_fidelity_prob)
                sparsity_list.append(Sparsity)

                count = count + 1
                print("count is", count)
                file_txt.write("count is" + str(format(count, '.3f')) + '\n')
                print("file_txt", file_txt)
                if count >= 2000:
                    break

            all_fidelity_acc = all_fidelity_acc_out / count
            all_infidelity_acc = all_infidelity_acc_out / count
            all_fidelity_prob = all_fidelity_prob_out / count
            all_infidelity_prob = all_infidelity_prob_out / count
            Sparsity = Sparsity_out / count

            # === 标准差 ===
            acc_mean, acc_std = self.compute_mean_std(acc_list)
            prob_mean, prob_std = self.compute_mean_std(prob_list)
            sparsity_mean, sparsity_std = self.compute_mean_std(sparsity_list)

            print("=================================================================")
            file_txt.write("=================================================================" + '\n')
            print("评价指标：指标忠诚度")
            file_txt.write("评价指标：指标忠诚度" + '\n')

            print("all_fidelity_acc:", format(all_fidelity_acc, '.3f'))
            file_txt.write("all_fidelity_acc:" + str(format(all_fidelity_acc, '.3f')) + '\n')

            print("all_infidelity_acc:", format(all_infidelity_acc, '.3f'))
            file_txt.write("all_infidelity_acc:" + str(format(all_infidelity_acc, '.3f')) + '\n')

            print("all_fidelity_prob:", format(all_fidelity_prob, '.3f'))
            file_txt.write("all_fidelity_prob:" + str(format(all_fidelity_prob, '.3f')) + '\n')

            print("all_infidelity_prob:", format(all_infidelity_prob, '.3f'))
            file_txt.write("all_infidelity_prob:" + str(format(all_infidelity_prob, '.3f')) + '\n')
            print("Sparsity:", format(Sparsity, '.3f'))
            file_txt.write("Sparsity:" + str(format(Sparsity, '.3f')) + '\n')

            print("all__fidelity_acc:", format(acc_mean, '.3f'), "±", format(acc_std, '.3f'))
            file_txt.write(
                "all_fidelity_acc:" + str(format(acc_mean, '.3f')) + "±" + str(format(acc_std, '.3f')) + '\n')

            print("all__fidelity_prob:", format(prob_mean, '.3f'), "±", format(prob_std, '.3f'))
            file_txt.write(
                "all_fidelity_prob:" + str(format(prob_mean, '.3f')) + "±" + str(format(prob_std, '.3f')) + '\n')

            print("Sparsity:", format(sparsity_mean, '.3f'), "±", format(sparsity_std, '.3f'))
            file_txt.write(
                "Sparsity:" + str(format(sparsity_mean, '.3f')) + "±" + str(format(sparsity_std, '.3f')) + '\n')

            file_txt.close()
            return

    def compute_mean_std(self, metric_list):
        """
        计算某个评价指标的均值和标准差
        :param metric_list: List[float]
        :return: (mean, std)
        """
        # tensor = torch.tensor(metric_list)
        tensor = torch.tensor(metric_list, dtype=torch.float32, device=self.dev)

        mean = torch.mean(tensor).item()
        std = torch.std(tensor, unbiased=True).item()
        return mean, std

    def lead_data(self):
        skeleton_name = self.arg.skeleton
        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
        print("Processing skeleton:", skeleton_name)
        print("Skeleton class:", action_class)
        self.action_class = action_class

        # skeleton_file = 'D:\\s\\st-gcn\\data\\NTU-RGB-D\\nturgb+d_skeletons\\'
        skeleton_file = './data/NTU-RGB-D/nturgb+d_skeletons/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = read_xyz(
            skeleton_file, max_body=max_body, num_joint=num_joint)

        self.model.eval()

        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data_model = data.float().to(self.dev)
        with torch.no_grad():
            out = self.model(data_model)
            label = out.argmax(dim=1, keepdim=True)
            probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
        N, C, T, V, M = data.size()
        x = data.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(N * T * V * M, C)
        return data_numpy, data, x, label, probability, skeleton_name

    def _set_feature_masks_(self, x):

        """
        该模块主要目的是获取全部特征的随机掩码。
        :param : x: 特定形式的骨架数据。
        :param : std: 方差。
        """
        std = 1
        if self.arg.feat_mask_type == 'individual_feature':
            N, F = x.size()
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__node_feat_mask__ = self.node_feat_mask

    def clear_masks(self):
        """
        该模块的目的是清除节点特征掩码
        """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__node_feat_mask__ = None
        self.node_feat_masks = None

    def calculation(self, data, label):

        """
        该模块主要目的是计算当前数据经过预训练模型对应label的判别概率。
        :param : data: 输入和掩膜乘积的骨架数据
        :param : label: 输入骨架动作的标签
        """
        self.model.eval()
        data = data.float().to(self.dev)
        with torch.no_grad():
            # out, _ = self.model(data)
            out = self.model(data)
            probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
        return probability

    def __loss__(self, masked_pred, original_pred, mask):

        """
        该模块为损失函数。
        :param : log_logits: 掩膜处理后的预测结果
        :param : pred_label: 骨架动作的标签
        """
        self.coeffs = coeffs
        size_loss = torch.sum(mask) * coeffs['mask_size']
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = coeffs['mask_ent'] * torch.mean(mask_ent_reg)
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        # loss = cce_loss + size_loss + mask_ent_loss
        loss = cce_loss

        return loss

    def forward(self, data, change_data, label):
        self.model.eval()
        self.clear_masks()
        # self._set_frame_mask_(data, label)
        mask = self.explainer(data, change_data, label)

        self.clear_masks()
        return mask

    # @torchsnooper.snoop()
    def explainer(self, data, change_data, label):
        change_data = change_data.to(self.dev)
        data = data.float().to(self.dev)
        self.model.eval()
        self.model.zero_grad()

        # <editor-fold desc="Get the initial prediction. 初次进行识别">
        with torch.no_grad():
            self.model.eval()
            # output, _ = self.model(data)
            output = self.model(data)
            prediction_label = output.argmax(dim=1, keepdim=True)
            # probability = (torch.nn.functional.softmax(output[0], dim=-1))[prediction_label].item()
        # 设置节点特征的初始掩膜
        N, F = change_data.size()
        nodemask = torch.nn.Parameter(torch.zeros(N, F).type(torch.float) * 1)
        # nodemask = self._set_feature_masks_(change_data)

        # 进行掩膜的学习和优化
        parameters = [nodemask]
        optimizer = torch.optim.Adam(parameters, lr=0.001)

        # 设置早停法
        bast_output = torch.tensor([0]).to(self.dev)
        index = 0
        for epoch in range(1, self.epochs + 1):
            self.model.eval()
            mask = (nodemask.sigmoid()).to(self.dev)

            h = (change_data * mask).to(self.dev)
            h = data_reshape(data=data, mask=h).float().to(self.dev)

            # out, _ = self.model(h)
            out = self.model(h)

            test_out = torch.nn.functional.softmax(out[0], dim=-1)
            # print(test_out[label])

            if test_out[label] > bast_output:
                bast_output = test_out[label]
                index = epoch
            else:
                if epoch - index > 5:
                    break

            loss = self.__loss__(out, prediction_label[0], mask=h)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # no fsc=======================================================

        # bast_feat_mask = mask.detach()
        bast_feat_mask = nodemask.detach()
        bast_feat_mask = bast_feat_mask.sigmoid()
        self.clear_masks()

        return bast_feat_mask

    def cff_GNNExplainer(self, data, change_data, label):
        change_data = change_data.to(self.dev)
        self.model = self.model.to(self.dev)
        data = data.float().to(self.dev)
        self.model.eval()
        self.model.zero_grad()
        # <初次进行识别">
        with torch.no_grad():
            self.model.eval()
            # output, _ = self.model(data)
            output = self.model(data)
            prediction_label = output.argmax(dim=1, keepdim=True)
        # 设置节点特征的初始掩膜
        N, F = change_data.size()
        # nodemask = torch.nn.Parameter(torch.zeros(N, F).type(torch.float) * 1)
        self._set_feature_masks_(change_data)
        nodemask = self.node_feat_mask
        nodemask = nodemask.detach().to(self.dev).requires_grad_(True)

        # 进行掩膜的学习和优化
        parameters = [nodemask]
        optimizer = torch.optim.Adam(parameters, lr=0.001)  # 0.001

        # 设置早停法
        bast_output = torch.tensor([0]).to(self.dev)
        index = 0
        # for epoch in range(1, self.epochs + 1):
        for epoch in range(1, 50 + 1):

            self.model.eval()
            mask = (nodemask.sigmoid()).to(self.dev)
            onemask = torch.ones_like(mask)
            unmask = onemask - mask
            # print(data_reshape(data=data, mask= self.node_feat_mask.sigmoid())[:, :, 0:29, :, :])
            # print(mask[2245:2255, :])
            h = (change_data * mask).to(self.dev)
            h = data_reshape(data=data, mask=h).float().to(self.dev)
            # out, _ = self.model(h)
            out = self.model(h)
            test_out = torch.nn.functional.softmax(out[0], dim=-1)  # 60个类别的置信度

            unh = (change_data * unmask).to(self.dev)
            unh = data_reshape(data=data, mask=unh).float().to(self.dev)
            # unout, _ = self.model(unh)
            unout = self.model(unh)
            untest_out = torch.nn.functional.softmax(unout[0], dim=-1)

            # print(test_out[label])

            if test_out[label] > bast_output:
                bast_output = test_out[label]
                index = epoch
            else:
                if epoch - index > 5:
                    break
            gam = 0.5
            lam = 1000
            alp = 0.6

            _, _, _, loss = self.cffloss(test_out, untest_out, output, gam, lam, alp, mask)
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            # print("epoch:",epoch,"optimizer nodemask",nodemask)
        bast_feat_mask = mask.detach()
        bast_feat_mask = bast_feat_mask.sigmoid()

        self.bast_feat_mask = bast_feat_mask
        self.clear_masks()

        return self.bast_feat_mask

    # @torchsnooper.snoop()
    def cffloss(self, pred1, pred2, ori_pred, gam, lam, alp, mask):  # loss_cfforg
        # self.args.gam = 0.5
        # self.args.lam = 1000
        # self.args.alp = 0.6
        # lam = 1
        # alp = 0.1
        # relu = torch.nn.ReLU()
        # relu = torch.nn.ELU(alpha=1.0) 
        relu = torch.nn.GELU()
        bpr1 = relu(gam + 0.5 - pred1)  # factual
        bpr2 = relu(gam + pred2 - 0.5)  # counterfactual
        L1 = torch.linalg.norm(mask, ord=1)
        loss = L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
        return bpr1, bpr2, L1, loss

    def evaluate(self, node_feature_mask, data, label, probability):
        return self.fidelity(node_feature_mask, data, label, probability)

    def fidelity(self, node_feature_mask, data, label, probability):
        file_txt = self.file_txt
        data = data.to(self.dev)
        # 获得初始数据全1的mask
        All_one_mask = torch.ones_like(data).to(self.dev)
        node_mask = data_reshape(data, node_feature_mask).to(self.dev)
        un_node_mask = (All_one_mask - node_mask).to(self.dev)

        print("模型识别的类别：", label, "   类别概率：", format(probability, '.4f'))
        file_txt.write("模型识别的类别：" + str(label) + "   类别概率：" + str(format(probability, '.4f')) + '\n')

        # 计算fidelity+ acc 和 fidelity+ prob
        self.model.eval()

        with torch.no_grad():
            # 使用节点特征
            All_in = (data * node_mask).float().to(self.dev)
            # All_out, _ = self.model(All_in)
            All_out = self.model(All_in)
            All_out = All_out.to(self.dev)
            # 记录节点特征的概率和标签
            All_label = All_out.argmax(dim=1, keepdim=True).item()
            All_prob = torch.nn.functional.softmax(All_out[0], dim=-1)[label].item()
            print("使用节点掩码类别：", All_label, "     使用节点掩码概率：", format(All_prob, '.4f'))
            file_txt.write(
                "使用节点掩码类别：" + str(All_label) + "   使用节点掩码概率：" + str(format(All_prob, '.4f')) + '\n')

            # 使用节点特征
            All_un_in = (data * un_node_mask).float().to(self.dev)
            # All_un_out, _ = self.model(All_un_in)
            All_un_out = self.model(All_un_in)
            # 记录节点特征的概率和标签
            All_un_label = All_un_out.argmax(dim=1, keepdim=True).item()
            All_un_prob = torch.nn.functional.softmax(All_un_out[0], dim=-1)[label].item()
            print("使用所有不重要节点掩码类别：", All_un_label, "     使用所有不重要节点掩码概率：",
                  format(All_un_prob, '.4f'))
            file_txt.write("使用所有不重要节点掩码类别：" + str(All_un_label) + "   使用所有不重要节点掩码概率：" + str(
                format(All_un_prob, '.4f')) + '\n')

        # 开始计算Fidelity_ACC
        acc_all = 1 if label == All_label else 0
        acc_un_all = 1 if label == All_un_label else 0
        # 开始计算prob
        prob_all = probability - All_prob
        prob_un_all = probability - All_un_prob

        # 所有不重要帧mask和节点特征的mask的正确率
        all_fidelity_acc = 1 - acc_un_all
        print("all_fidelity_acc:", all_fidelity_acc)
        file_txt.write("all_fidelity_acc:" + str(all_fidelity_acc) + '\n')
        # 所有重要帧mask和节点特征的mask的正确率
        all_infidelity_acc = 1 - acc_all
        print("all_infidelity_acc:", all_infidelity_acc)
        file_txt.write("all_infidelity_acc:" + str(all_infidelity_acc) + '\n')
        # 所有不节点特征的mask的正确率
        all_fidelity_prob = prob_un_all
        print("all_fidelity_prob:", format(all_fidelity_prob, '.4f'))
        file_txt.write("all_fidelity_prob:" + str(all_fidelity_prob) + '\n')
        # 所有重要节点特征的mask的正确率
        all_infidelity_prob = prob_all
        print("all_infidelity_prob:", format(all_infidelity_prob, '.4f'))
        file_txt.write("all_infidelity_prob:" + str(all_infidelity_prob) + '\n')
        # 计算Sparsity
        Sparsity = torch.sum(node_mask) / torch.sum(All_one_mask).item()
        print("Sparsity：", format(Sparsity, '.4f'))
        file_txt.write("Sparsity：" + str(format(Sparsity, '.4f')) + '\n')

        return all_fidelity_acc, all_infidelity_acc, \
            all_fidelity_prob, all_infidelity_prob, Sparsity

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Dome for ST-GCN Explainer')
        parser.add_argument('--skeleton',
                            default='S015C001P008R001A013',
                            help='Path to video')
        # work4
        # self.skeleton = "S015C001P008R001A013,S016C001P021R001A039,S027C001P008R002A113"
        # work3
        # ntu60: S005C001P016R001A001 , S012C001P037R001A021 , S002C002P007R001A038 ,S009C003P015R002A040
        # ntu120: S025C002P059R001A118 ,S023C002P065R001A061 , S027C001P081R002A076 ,S022C001P066R002A068
        parser.add_argument('--plot_action',
                            default=True,
                            help='save action as image',
                            type=bool)
        parser.add_argument('--output_dir',
                            # default='./data/demo_skeleton/result',
                            default='./work3result_v/',
                            help='Path to save results')
        parser.add_argument('--run_device',
                            default='cuda:2',
                            help='Dev to use, e.g., "cuda:0", "cuda:1", "cuda:2" or "cpu"')
        parser.add_argument('--exp_type',
                            default='cff',  # cff,gnnexplainer
                            help='one of gnn_explainer,pg_explainer,GNN_Explainer,cff_GNNExplainer')
        parser.add_argument('--data_level',
                            default='instance',
                            help='instance or test_set')
        parser.add_argument('--valid',
                            default='xview',
                            help='One of xsub and xview,and csub csetup,kinetics-skeleton')
        parser.add_argument('--feat_mask_type',
                            default='individual_feature',
                            help='individual_feature or cam')
        # parser.add_argument('--pacing', type=int, default=30, nargs='+',
        #                     help='how mach frame data you want one time')
        # parser.add_argument('--stride', type=int, default=10, nargs='+',
        #                     help='how mach stride you want before two frame data')
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')

        args = parser.parse_known_args(namespace=parent_parser)
        # print("mark", parent_parser.valid)
        # parser.set_defaults(
        #     config='./config/st_gcn/{}/test.yaml'.format(parent_parser.valid))
        ntu60_valids = ['xsub', 'xview']
        valid = args[0].valid
        if valid in ntu60_valids:
            config_path = './config/st_gcn/ntu-{}/test.yaml'.format(parent_parser.valid)  # cff,gnnexplainer
        else:
            config_path = './config/st_gcn/ntu120-{}/test.yaml'.format(parent_parser.valid)
        parser.set_defaults(config=config_path)
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
