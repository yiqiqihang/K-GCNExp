import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import time
import os
import numpy as np
from numpy import *
# from .processor import Processor
from .recognition import REC_Processor
from math import sqrt
from torch.nn.functional import cross_entropy
import tools.utils as utils
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new, plucker
from torch_geometric.nn import MessagePassing
from tools.utils.visualization_skeleton import plot_action
from torchlight import str2bool
import csv
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.integrate import simps
import torchsnooper
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.container")
import random
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


class Explainer(REC_Processor):
    def __init__(self, argv=None):
        super().__init__(argv)
        # self.epochs = 100
        self.epochs = 50
        self.print_log = True

    def start(self):
        print("Using {} weights.".format(self.arg.valid))
        explainer_dict = {'explainer': self.explainer,
                          'GNN_explainer': self.GNN_Explainer,
                          'pg_explainer': self.PG_Explainer}
        self.method = self.arg.exp_type.lower()
        out = self.process_exps(explainer_dict[self.method])
        self.dev = 'cuda:2'

        return
    # @torchsnooper.snoop()
    def process_exps(self, exp_func):
        """
        该模块主要是设置模型解释的样本，分为单实例和多个实例。
        instance：单实例 定性解决问题时使用。
        test_set：多实例 定量分析效果。
        :param :exp_func 所选取的可解释方法。
        """
        #记录结果
        time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
        self.dev = 'cuda:2'
        file_txt = self.file_txt = open(f'result-forcam-20240325/GNN_explainer-{self.arg.valid}-{time_now}.txt', mode='a', encoding="utf-8")
        if self.arg.data_level == 'instance':
            label_name_path = '/data/home/yr/YRR/HDGCN/data/label_name.txt'
            with open(label_name_path) as f:
                label_name = f.readlines()
                label_name = [line.rstrip() for line in label_name]
                self.label_name = label_name
            skeleton_name = self.arg.skeleton
            data_numpy, data, x, label, probability, skeleton_name = self.lead_data()
            print("Processing skeleton:", skeleton_name)
            visoutput_result_dir =f'./result-forcam-20240325/instance-visforgnne/{self.method}-ntu-{self.arg.valid}/'
            output_result_dir = '{}-{}/{}'.format(visoutput_result_dir,time_now, skeleton_name)
            
            if not os.path.exists('{}/cam/'.format(output_result_dir)):
                os.makedirs('{}/cam/'.format(output_result_dir))

            skeleton_file = '/data/home/yr/YRR/HDGCN/data/nturgbd_raw/ntuall/'#ntu120all
            skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
            data_numpy = read_xyz(skeleton_file, max_body=max_body, num_joint=num_joint)
            node_feature_mask = exp_func(data, x, label,skeleton_name)

            if self.arg.plot_action:
                self.num_node = 25
                self_link = [(i, i) for i in range(self.num_node)]
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                    (22, 23), (23, 8), (24, 25), (25, 12)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]  #序号从1开始，故需要 -1，所有相连的元组
                edge = self_link + neighbor_link
                C, T, V, M = data_numpy.shape
                node_mask = data_reshape(data, node_feature_mask)
                wise = torch.sum(node_mask, dim=1)
                # wise = wise.unsqueeze(0)
                plot_action(data_numpy,edge, wise.cpu().numpy(), save_dir=output_result_dir,
                                                        save_type=self.arg.exp_type)
            if self.arg.data_level == 'instance':
                all_fidelity_acc,  all_infidelity_acc, \
                    all_fidelity_prob,  all_infidelity_prob,Sparsity = \
                        self.evaluate(node_feature_mask, data,  label, probability)
            return 
        
        if self.arg.data_level == 'test_set':

            count = 0
            all_fidelity_acc_out = 0
            all_infidelity_acc_out = 0
            all_fidelity_prob_out = 0
            all_infidelity_prob_out = 0
            Sparsity_out = 0
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
            #         print("Processing skeleton:", file)
            #         file_txt.write("Processing skeleton:" + file)
            loader = self.data_loader['test']
            for data, label,index in loader:
                self.model.eval()
                self.model = self.model.to(self.dev)
                # data = torch.from_numpy(data_numpy)
                # data = data.unsqueeze(0)
                data_model = data.float().to(self.dev)
                with torch.no_grad():
                    out,_= self.model(data_model)
                    label = out.argmax(dim=1, keepdim=True).item()
                    probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
                N, C, T, V, M = data.size()
                x = data.permute(0, 2, 3, 4, 1).contiguous()
                x = x.view(N * T * V * M, C)
         

                # # output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_file)
                # # if not os.path.exists('{}'.format(output_result_dir)):
                # #     os.makedirs('{}'.format(output_result_dir))
                # visoutput_result_dir = '/data/home/yr/YRR/gcne/result231228/gnnexplainervisadmmgeo/gnnexplainervisadmmgeo'
                # # output_result_dir = '{}/{}'.format(visoutput_result_dir, skeleton_file)
                # output_result_dir = '{}-{}/{}'.format(visoutput_result_dir,time_now, file)
                # if not os.path.exists('{}'.format(output_result_dir)):
                #     os.makedirs('{}'.format(output_result_dir))
                
                node_feature_mask = exp_func(data, x, label)

                #if self.arg.plot_action:
                #    node_mask = data_reshape(data, node_feature_mask)
                #    wise = torch.sum(node_mask, dim=1)
                #    utils.visualization_skeleton.plot_action(data_numpy, self.model.graph.edge, wise.numpy(),
                #                                             save_dir=output_result_dir,
                #                                             save_type=self.arg.exp_type)

                all_fidelity_acc,  all_infidelity_acc, \
                all_fidelity_prob,  all_infidelity_prob,Sparsity = \
                    self.evaluate(node_feature_mask, data,  label, probability)
                
                all_fidelity_acc_out += all_fidelity_acc                 
                all_infidelity_acc_out += all_infidelity_acc               
                all_fidelity_prob_out += all_fidelity_prob                
                all_infidelity_prob_out += all_infidelity_prob
                Sparsity_out += Sparsity

                count = count + 1
                print("count is", count)
                if count >= 50:
                    all_fidelity_acc = all_fidelity_acc_out / count
                    all_fidelity_prob = all_fidelity_prob_out / count
                    Sparsity = Sparsity_out/count
                    print("current all_un_screw_node_fidelity_acc", all_fidelity_acc)                  
                    print("current all_un_screw_node_fidelity_prob", all_fidelity_prob)            
                    print("current Sparsity", Sparsity)

                if count >= 2000:
                    break

            
            all_fidelity_acc = all_fidelity_acc_out / count            
            all_infidelity_acc = all_infidelity_acc_out / count           
            all_fidelity_prob = all_fidelity_prob_out / count           
            all_infidelity_prob = all_infidelity_prob_out / count
            Sparsity = Sparsity_out/count
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
            file_txt.close()
            return

    def lead_data(self):
        skeleton_name = self.arg.skeleton
        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
        print("Processing skeleton:", skeleton_name)
        print("Skeleton class:", action_class)
        self.action_class = action_class

        # skeleton_file = 'D:\\s\\st-gcn\\data\\NTU-RGB-D\\nturgb+d_skeletons\\'
        skeleton_file = '/data/home/yr/YRR/HDGCN/data/nturgbd_raw/ntuall/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = read_xyz(
            skeleton_file, max_body=max_body, num_joint=num_joint)

        self.model.eval()

        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data_model = data.float().to(self.dev)
        self.model = self.model.to(self.dev)
        with torch.no_grad():
            out,_ = self.model(data_model)
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
            out,_ = self.model(data)
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

        # loss = cce_loss + size_loss + mask_ent_loss#hdgcn_st-nan
        loss = cce_loss

        return loss

    def forward(self, data, change_data, label):
        self.model.eval()
        self.clear_masks()
        #self._set_frame_mask_(data, label)
        mask = self.explainer(data, change_data, label)

        self.clear_masks()
        return mask
    # @torchsnooper.snoop()
    def explainer(self, data, change_data, label):
        #N, C, T, V, M = frame_mask.size()
        #x = frame_mask.permute(0, 2, 3, 4, 1).contiguous()
        #x = x.view(N * T * V * M, C)
        # print(x.size())
        # print(x[2245:2255, :])
        # print(x.sum())
        change_data = change_data.to(self.dev)
        data = data.float().to(self.dev)
        self.model.eval()
        self.model.zero_grad()

        # <editor-fold desc="Get the initial prediction. 初次进行识别">
        with torch.no_grad():
            self.model.eval()
            output,_ = self.model(data)
            prediction_label = output.argmax(dim=1, keepdim=True)
            # probability = (torch.nn.functional.softmax(output[0], dim=-1))[prediction_label].item()

        # </editor-fold>
        # #with fsc=======================================================
        # time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # svalid = self.arg.valid
        # directory=f'result231228/gnnexplainer_FSC/gnnexplainer_FSC_20240202_{svalid}'
        # csv_file_name = skeletonfile + time_now + '.csv'
        # csv_file_path = os.path.join(directory, csv_file_name)
        # AUFSC_file_name = 'AUFSC' + skeletonfile + time_now + '.csv'
        # AUFSC_file_path = os.path.join(directory, AUFSC_file_name)
        # # 检查文件夹是否存在，如果不存在，则创建
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        

        # # 检查文件是否为空
        # if not os.path.exists(AUFSC_file_path) or os.stat(AUFSC_file_path).st_size == 0:
        #     with open(AUFSC_file_path, mode='w', encoding="utf-8", newline='') as aufile:
        #         # 如果文件是新的或为空，则添加头部，例如：
        #         aufile.write('skeletonname,AUFSC\n')
        
        # if not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0:
        #     with open(csv_file_path, mode='w', encoding="utf-8", newline='') as file:
        #         # 如果文件是新的或为空，则添加头部，例如：
        #         file.write('Fidelity,Sparsity\n')
        # # 设置节点特征的初始掩膜
        # N, F = change_data.size()
        # nodemask = torch.nn.Parameter(torch.zeros(N, F).type(torch.float) * 1)
        # # nodemask = self._set_feature_masks_(change_data)

        # # 进行掩膜的学习和优化
        # parameters = [nodemask]
        # optimizer = torch.optim.Adam(parameters, lr=0.001)

        # # 设置早停法
        # bast_output = torch.tensor([0]).to(self.dev)
        # index = 0
        # with open(csv_file_path, mode='a', newline='', encoding="utf-8") as file:
        #     writer = csv.writer(file)
        #     for epoch in range(1, self.epochs + 1):
        #         self.model.eval()
        #         mask = (nodemask.sigmoid()).to(self.dev)
        #         # print(data_reshape(data=data, mask= self.node_feat_mask.sigmoid())[:, :, 0:29, :, :])
        #         # print(mask[2245:2255, :])
        #         h = (change_data * mask).to(self.dev)
        #         h = data_reshape(data=data, mask=h).float().to(self.dev)

        #         out = self.model(h)

        #         test_out = torch.nn.functional.softmax(out[0], dim=-1)
        #         # print(test_out[label])

        #         if test_out[label] > bast_output:
        #             bast_output = test_out[label]
        #             index = epoch
        #         else:
        #             if epoch - index > 5:
        #                 break


        #         all_fidelity_acc,  all_infidelity_acc, \
        #         all_fidelity_prob,  all_infidelity_prob,Sparsity = \
        #             self.evaluate(mask.sigmoid(), data, label, probability)
        #         sparsity_value = Sparsity.item()
        #         writer.writerow([all_fidelity_prob, sparsity_value])
        #         loss = self.__loss__(out, prediction_label[0], mask=h)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        # ## 绘制fsc折线图
        # # 读取CSV文件
        # filename = file.name
        # df1 = pd.read_csv(filename)
        
        # # 绘制折线图
        # fig, ax = plt.subplots(figsize=(8, 6))
        
        # # 使用样条插值函数平滑曲线
        # fidelity = df1['Fidelity']
        # sparsity = df1['Sparsity']
        # # 对数据进行排序
        # sorted_indices = sparsity.argsort()
        # sorted_sparsity = sparsity[sorted_indices]
        # fsorted_indices = fidelity.argsort()
        # sorted_fidelity = fidelity[fsorted_indices]
  
        # # sorted_fidelity = fidelity[sorted_indices]
        # # x_new = sorted_sparsity
        # # y_new = make_interp_spline(sorted_sparsity, fidelity)(x_new)
        # ax.plot(sorted_sparsity, sorted_fidelity, linestyle='-', color='dodgerblue', label='Ours')
        # # ax.plot(df1['Sparsity'], df1['Fidelity'], marker='o', linestyle='-', color='blue', label='Ours')
        # ax.set_title('Fidelity Sparsity Curve')
        # ax.set_xlabel('Sparsity')
        # ax.set_ylabel('Fidelity')
        # ax.legend()
        # ax.grid(True)

        # print("saveing")
        # plt.savefig('{}/{}{}.svg'.format(directory, skeletonfile,time_now))
        # plt.savefig('{}/{}{}.png'.format(directory, skeletonfile,time_now))
        # plt.savefig('{}/{}{}.pdf'.format(directory, skeletonfile,time_now))
        
        # with open(AUFSC_file_path, mode='a', newline='', encoding="utf-8") as aufile:
        #     aufscwriter = csv.writer(aufile)
        #     # 计算面积aufsc
        #     fscarea = simps(sorted_fidelity, sorted_sparsity)  
        #     aufscwriter.writerow([skeletonfile, fscarea])              
        #     print("AUFSC面积为：", fscarea)

        #with fsc=======================================================
        #no fsc=======================================================  
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
            # print(data_reshape(data=data, mask= self.node_feat_mask.sigmoid())[:, :, 0:29, :, :])
            # print(mask[2245:2255, :])
            h = (change_data * mask).to(self.dev)
            h = data_reshape(data=data, mask=h).float().to(self.dev)

            out,_ = self.model(h)

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
        #no fsc=======================================================   

        bast_feat_mask = mask.detach()
        bast_feat_mask = bast_feat_mask.sigmoid()
        self.clear_masks()

        return bast_feat_mask

    def PG_Explainer(self):
        return

    def GNN_Explainer(self):
        return

    def evaluate(self, node_feature_mask, data,  label, probability):
        return self.fidelity( node_feature_mask, data,  label, probability)

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
            All_out,_ = self.model(All_in)
            All_out=All_out.to(self.dev)
            # 记录节点特征的概率和标签
            All_label = All_out.argmax(dim=1, keepdim=True).item()
            All_prob = torch.nn.functional.softmax(All_out[0], dim=-1)[label].item()
            print("使用节点掩码类别：", All_label, "     使用节点掩码概率：", format(All_prob, '.4f'))
            file_txt.write("使用节点掩码类别：" + str(All_label) + "   使用节点掩码概率：" + str(format(All_prob, '.4f')) + '\n')

            
            # 使用节点特征
            All_un_in = (data *  un_node_mask).float().to(self.dev)
            All_un_out,_  = self.model(All_un_in)
            All_un_out = All_un_out.to(self.dev)
            # 记录节点特征的概率和标签
            All_un_label = All_un_out.argmax(dim=1, keepdim=True).item()
            All_un_prob = torch.nn.functional.softmax(All_un_out[0], dim=-1)[label].item()
            print("使用所有不重要节点掩码类别：", All_un_label, "     使用所有不重要节点掩码概率：", format(All_un_prob, '.4f'))
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
        Sparsity = torch.sum(node_mask)/torch.sum(All_one_mask).item()
        print("Sparsity：", format(Sparsity, '.4f'))
        file_txt.write("Sparsity：" + str(format(Sparsity, '.4f')) + '\n')


        if self.arg.data_level == 'instance':
            current_sparsity = Sparsity
            sparsity_step = 0.05
            # 目标 Sparsity
            target_sparsity = 0.35
            while target_sparsity <= current_sparsity: 

                # # 方法一          
                # # 计算需要调整的比例
                # adjustment_factor = (current_sparsity-target_sparsity )/current_sparsity
                # # 调整 mask 的值
                # new_mask = screw_all_mask*node_mask * adjustment_factor
                # # print( new_mask.shape)

                # 方法二        
                new_mask = node_mask
                # print(new_mask)
                # 计算需要调整的比例
                adjustment_factor = target_sparsity/current_sparsity
                # 设置随机种子以确保可重复性
                random.seed(42)
                # 计算 new_mask 中的元素总数
                total_elements = new_mask.numel()
                # 计算需要置为零的元素数量，即总元素数的1-adjustment_factor
                num_zeros = int(total_elements * (1-adjustment_factor.item()))
                # 生成一个包含需要置为零的元素索引的随机列表
                zero_indices = random.sample(range(total_elements), num_zeros)
                # 将这些随机索引对应的元素值设置为零
                new_mask.view(-1)[zero_indices] = 0
                # print( new_mask.shape)
                # ttt
                new_un_mask = (All_one_mask - new_mask).to(self.dev) #torch.Size([1, 3, 68, 25, 2])
                
                #只使用1-saliency_map
                new_un_mask_in = (data*new_un_mask).float().to(self.dev)
                # new_un_mask_in = new_un_mask_in.double() # #64 for gradcampps
                new_un_mask_out,_  = self.model(new_un_mask_in)
                new_un_mask_out= new_un_mask_out.to(self.dev)
                new_un_mask_prob = torch.nn.functional.softmax(new_un_mask_out[0], dim=-1)[label].item()
                # 计算Fidelity_prob
                # prob_mask = probability - mask_prob   
                new_prob_un_mask = probability - new_un_mask_prob
                New_sparsity = 1 - torch.sum(new_mask) / torch.sum(All_one_mask).item()
                print(f"Target Sparsity: {target_sparsity}, New Unmask Probability: {new_prob_un_mask}")
                file_txt.write("开始计算FSC需要要数据!!!!!!!!" + '\n')
                file_txt.write("Target Sparsity: " + str(format(target_sparsity, '.4f')) + '\n')
                file_txt.write("New Unmask Probability: " + str(format(new_prob_un_mask, '.4f')) + '\n')
                print(f"New Sparsity: {New_sparsity}")
                file_txt.write("New Sparsity: " + str(format(New_sparsity, '.4f')) + '\n')    
                print(file_txt)
                print(file_txt)
                target_sparsity += sparsity_step

        return all_fidelity_acc,  all_infidelity_acc, \
               all_fidelity_prob, all_infidelity_prob, Sparsity
    
    def str2bool(self,v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = REC_Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Dome for HD-GCN Explainer')
        parser.add_argument('--skeleton',
                            default='S005C001P016R001A001',
                            help='Path to video')
        #S025C002P059R001A118
        parser.add_argument('--plot_action',
                           default=False,
                           help='save action as image',
                           type=bool)
        parser.add_argument('--output_dir',
                            default='./work3result_v/',
                            help='Path to save results')
        parser.add_argument('--exp_type',
                            default='Explainer',
                            help='one of gnn_explainer,pg_explainer,GNN_Explainer')
        parser.add_argument('--data_level',
                            default='test_set',
                            help='instance or test_set')
        parser.add_argument('--valid',
                            default='ntu-xview',
                            help='One of, ntu-xsub , ntu-xview, \
                             ntu120-csub, ntu120-csetup')
        parser.add_argument('--feat_mask_type',
                            default='individual_feature',
                            help='individual_feature or cam')
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # print("mark", parent_parser.valid)
        # parser.set_defaults(
        #     # config='./config/st_gcn/ntu-{}/test.yaml'.format(parent_parser.valid))
        #     config='./config/{}/bone_com_1test.yaml'.format(parent_parser.valid))
        args = parser.parse_known_args(namespace=parent_parser)
        parser.set_defaults(config='./config/{}/bone_com_1_test.yaml'.format(parent_parser.valid))
        # parser.add_argument('--print_log', default=True, help='Help message for print_log')
        # endregion yapf: enable
        # parser.add_argument( '--print-log', type=str2bool,default=True, help='print logging or not')
        parser.set_defaults(print_log=False)
        return parser
