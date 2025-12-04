import scipy.sparse as sp
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import time
import os
import numpy as np
import tools.utils as utils
from numpy import *
from .processor import Processor
from math import sqrt
from torch.nn.functional import cross_entropy
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new
from torch_geometric.nn import MessagePassing
from net.utils.graph import Graph

reg_coefs = (0.05, 1.0)

max_body = 2
num_joint = 25
max_frame = 300

# 记录结果
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
file_txt = open('result/' + time_now + '.txt', mode='a', encoding="utf-8")


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


class GNNExplainer(Processor):

    def __init__(self, argv=None):
        super().__init__(argv)
        self.graph = Graph('ntu-rgb+d')
        self.epochs = 100

    def start(self):
        print("Using {} weights.".format(self.arg.valid))
        explainer_dict = {'explain': self.explain,
                          'pg_explainer': self.PG_Explainer}
        mothod = self.arg.exp_type.lower()
        out = self.process_exps(explainer_dict[mothod])

        return

    def process_exps(self, exp_func):

        if self.arg.data_level == 'instance':
            label_name_path = './resource/ntu_skeleton/label_name.txt'
            with open(label_name_path) as f:
                label_name = f.readlines()
                label_name = [line.rstrip() for line in label_name]
                self.label_name = label_name

            data_numpy, data, x, label, probability, skeleton_name = self.lead_data()
            node_feature_mask = exp_func(data, x, label)
            data = data.to(self.dev)
            output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_name)
            if not os.path.exists('{}/cam/'.format(output_result_dir)):
                os.makedirs('{}/cam/'.format(output_result_dir))

            if self.arg.plot_action:
                node_mask = data_reshape(data, node_feature_mask)
                wise = torch.sum(node_mask, dim=1)
                # wise = wise.unsqueeze(0)
                print("画图")
                utils.visualization_skeleton.plot_action(data_numpy,self.model.graph.edge, wise.numpy(), save_dir=output_result_dir,
                                                         save_type=self.arg.exp_type)

            return self.fidelity(node_feature_mask, data, x, label, probability)

        if self.arg.data_level == 'test_set':

            count = 0
            fidelity_acc_out = 0
            infidelity_acc_out = 0
            fidelity_prob_out = 0
            infidelity_prob_out = 0
            skeleton_file_path = './data/NTU-RGB-D/nturgb+d_skeletons/'
            for root, dirs, files in os.walk(skeleton_file_path):
                for file in files:
                    print("=================================================================")
                    file_txt.write("=================================================================" + '\n')
                    file_txt.write(str(datetime.datetime.now()) + '\n')
                    skeleton_file = skeleton_file_path + file

                    print("Processing skeleton:", file)
                    file_txt.write("Processing skeleton:" + file + '\n')

                    data_numpy = read_xyz(
                        skeleton_file, max_body=max_body, num_joint=num_joint)
                    self.model.eval().to(self.dev)

                    data = torch.from_numpy(data_numpy)
                    data = data.unsqueeze(0).to(self.dev)
                    data_model = data.float().to(self.dev)
                    with torch.no_grad():
                        out = self.model(data_model)
                        label = out.argmax(dim=1, keepdim=True).item()
                        probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
                    N, C, T, V, M = data.size()
                    x = data.permute(0, 2, 3, 4, 1).contiguous()
                    x = x.view(N * T * V * M, C).to(self.dev)
                    node_feature_mask = exp_func(data, x, label)
                    fidelity_acc, infidelity_acc, fidelity_prob, infidelity_prob = \
                        self.evaluate(node_feature_mask, data, x, label, probability)
                    fidelity_acc_out += fidelity_acc
                    infidelity_acc_out += infidelity_acc
                    fidelity_prob_out += fidelity_prob
                    infidelity_prob_out += infidelity_prob

                    count = count + 1
                    print("count",count)
                    if count >= 2000:
                        break

                fidelity_acc = fidelity_acc_out / count
                infidelity_acc = infidelity_acc_out / count
                fidelity_prob = fidelity_prob_out / count
                infidelity_prob = infidelity_prob_out / count
                print("=================================================================")
                file_txt.write("=================================================================" + '\n')
                print("评价指标：指标忠诚度")
                file_txt.write("评价指标：指标忠诚度" + '\n')
                print("fidelity_acc:", format(fidelity_acc, '.3f'))
                file_txt.write("fidelity_acc:" + str(format(fidelity_acc, '.3f')) + '\n')
                print("infidelity_acc:", format(infidelity_acc, '.3f'))
                file_txt.write("infidelity_acc:" + str(format(infidelity_acc, '.3f')) + '\n')
                print("fidelity_prob:", format(fidelity_prob, '.3f'))
                file_txt.write("fidelity_prob:" + str(format(fidelity_prob, '.3f')) + '\n')
                print("infidelity_prob:", format(infidelity_prob, '.3f'))
                file_txt.write("infidelity_prob:" + str(format(infidelity_prob, '.3f')) + '\n')
                print("method:", "gnnexplainer")
                print(self.arg.valid)
                file_txt.close()
            return

    def lead_data(self):
        skeleton_name = self.arg.skeleton
        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
        print("Processing skeleton:", skeleton_name)
        print("Skeleton class:", action_class)

        # skeleton_file = './data/NTU-RGB-D/nturgb+d_skeletons/'
        skeleton_file = './data/NTU-RGB-D/ntuall/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = read_xyz_new(
            skeleton_file, max_body=max_body, num_joint=num_joint)

        self.model.eval().to(self.dev)

        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data_model = data.float().to(self.dev)
        with torch.no_grad():
            out = self.model(data_model).to(self.dev)
            label = out.argmax(dim=1, keepdim=True).to(self.dev)
            probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
        N, C, T, V, M = data.size()
        x = data.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(N * T * V * M, C)
        return data_numpy, data, x, label, probability, skeleton_name

    def get_edge_index(self):
        # load graph
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = A.squeeze(0)
        edge_index_temp = sp.coo_matrix(A)
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
        edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式

        return edge_index_A

    def _set_masks(self, x, edge_index):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """
        (N, F), E = x.size(), edge_index.size(1)

        # std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        std = 3
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def _loss(self, masked_pred, original_pred, edge_mask):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = 0.05
        entropy_reg = 1.0
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def forward(self, data, change_data, label):
        self.model.eval()
        self._clear_masks()
        self._set_masks(change_data)
        mask = self.explain(data, change_data, label)

        self._clear_masks()
        return mask

    def explain(self, data, change_data, label):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param data: Current explanaiton
        :param change_data: Current explanaiton
        :param label: Current explanaiton
        :return: explanation graph and edge weights
        """
        change_data = change_data.to(self.dev)
        data = data.float().to(self.dev)

        # Prepare model for new explanation run
        self.model.eval()
        self.model.zero_grad()
        self._clear_masks()

        # # Get the initial prediction.
        with torch.no_grad():
            self.model.eval()
            output = self.model(data)
            prediction_label = output.argmax(dim=1, keepdim=True)

        # 设置节点特征的初始掩膜
        edge_index = self.get_edge_index()
        self._set_masks(change_data, edge_index)

        # 进行掩膜的学习和优化
        parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=0.1)

        # 设置早停法
        bast_output = torch.tensor([0]).to(self.dev)
        index = 0

        for epoch in range(1, self.epochs + 1):
            self.model.eval()
            mask = self.node_feat_mask.sigmoid().to(self.dev)
            # print(mask)
            h = (change_data * mask).to(self.dev)

            h = data_reshape(data=data, mask=h).float()
            out = self.model(h)
            test_out = torch.nn.functional.softmax(out[0], dim=-1).to(self.dev)

            loss = self._loss(out, prediction_label[0], h)

            if test_out[label] > bast_output:
                bast_output = test_out[label]
                index = epoch
            else:
                if epoch - index > 5:
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bast_feat_mask = self.node_feat_mask.detach().sigmoid()

            self._clear_masks()

        return bast_feat_mask

    def fidelity(self, node_feature_mask, data, x, label, probability):

        # <editor-fold desc="获得初始数据全1的mask">
        All_one_mask = torch.ones_like(data).to(self.dev)
        node_mask = data_reshape(data, node_feature_mask).to(self.dev)
        un_node_mask = (All_one_mask - node_mask).to(self.dev)
        # </editor-fold>

        # <editor-fold desc="">
        print("模型识别的类别：", label, "   类别概率：", format(probability, '.4f'))
        file_txt.write("模型识别的类别：" + str(label) + "   类别概率：" + str(format(probability, '.4f')) + '\n')
        # </editor-fold>

        # 计算fidelity+ acc 和 fidelity+ prob
        self.model.eval()

        with torch.no_grad():
            # 使用节点重要掩码
            importance_node_mask = (data * node_mask).float().to(self.dev)
            node_mask_out = self.model(importance_node_mask).to(self.dev)
            # 记录概率和标签
            node_mask_label = node_mask_out.argmax(dim=1, keepdim=True).item()
            node_mask_prob = torch.nn.functional.softmax(node_mask_out[0], dim=-1)[label].item()
            print("使用重要节点特征掩码类别：", node_mask_label, "     使用重要节点特征掩码概率：", format(node_mask_prob, '.4f'))
            file_txt.write(
                "使用重要节点特征掩码类别：" + str(node_mask_label) + "   使用重要节点特征掩码概率：" + str(format(node_mask_prob, '.4f')) + '\n')

            # 使用节点不重要掩码
            unimportance_node_mask = (data * un_node_mask).float().to(self.dev)
            un_node_mask_out = self.model(unimportance_node_mask).to(self.dev)
            # 记录概率和标签
            un_node_mask_label = un_node_mask_out.argmax(dim=1, keepdim=True).item()
            un_node_mask_prob = torch.nn.functional.softmax(un_node_mask_out[0], dim=-1)[label].item()
            print("使用不重要节点特征掩码类别：", un_node_mask_label, "     使用不重要节点特征掩码概率：", format(un_node_mask_prob, '.4f'))
            file_txt.write("使用不重要节点特征掩码类别：" + str(un_node_mask_label) + "   使用不重要节点特征掩码概率：" + str(
                format(un_node_mask_prob, '.4f')) + '\n')

        # 开始计算Fidelity_ACC
        acc_node = 1 if label == node_mask_label else 0
        acc_un_node = 1 if label == un_node_mask_label else 0

        # 开始计算prob
        prob_node = probability - node_mask_prob
        prob_un_node = probability - un_node_mask_prob

        # 重要节点特征mask的正确率
        fidelity_acc = 1 - acc_un_node
        print("fidelity_acc:", fidelity_acc)
        file_txt.write("fidelity_acc:" + str(fidelity_acc) + '\n')
        # 不重要节点特征的mask的正确率
        infidelity_acc = 1 - acc_node
        print("infidelity_acc:", infidelity_acc)
        file_txt.write("infidelity_acc:" + str(infidelity_acc) + '\n')

        # 只有不重要帧mask的概率
        fidelity_prob = prob_un_node
        print("fidelity_prob:", format(fidelity_prob, '.4f'))
        file_txt.write("fidelity_prob:" + str(fidelity_prob) + '\n')
        # 所有不重要帧mask和节点特征的mask的正确率
        infidelity_prob = prob_node
        print("infidelity_prob:", format(infidelity_prob, '.4f'))
        file_txt.write("infidelity_prob:" + str(infidelity_prob) + '\n')

        return fidelity_acc, infidelity_acc, fidelity_prob, infidelity_prob

    def evaluate(self, node_feature_mask, data, x, label, probability):
        return self.fidelity(node_feature_mask, data, x, label, probability)

    def PG_Explainer(self):
        return

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Dome for ST-GCN Explainer')
        parser.add_argument('--skeleton',
                            default='S029C002P080R002A100',
                            help='Path to video')
        parser.add_argument('--plot_action',
                            default=True,
                            help='save action as image',
                            type=bool)
        parser.add_argument('--output_dir',
                            default='./work3result_v/',
                            help='Path to save results')
        parser.add_argument('--exp_type',
                            default='explain',
                            help='one of gnn_explainer,pg_explainer')
        parser.add_argument('--data_level',
                            default='instance',
                            help='instance or test_set')
        parser.add_argument('--valid',
                            default='csub',
                            help='One of xsub and xview')
        parser.add_argument('--mask_type',
                            default='random',
                            help='one of cam or random')
        parser.add_argument('--feat_mask_type',
                            default='individual_feature',
                            help='individual_feature or cam')
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')

        args = parser.parse_known_args(namespace=parent_parser)
        # print("mark", parent_parser.valid)
        parser.set_defaults(
            config='./config/st_gcn/ntu120-{}/test.yaml'.format(parent_parser.valid))
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
