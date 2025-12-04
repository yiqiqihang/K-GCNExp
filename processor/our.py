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
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new
import tools.utils as utils
from torch_geometric.nn import MessagePassing

# 记录结果
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
file_txt = open('result/' + time_now + '.txt', mode='a', encoding="utf-8")

max_body = 2
num_joint = 25
max_frame = 300

coeffs = {
    'mask_size': 0.05,
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
        self.epochs = 100

    def start(self):
        print("Using {} weights.".format(self.arg.valid))
        explainer_dict = {'explainer': self.explainer}
        method = self.arg.exp_type.lower()
        print("我们的代码")
        out = self.process_exps(explainer_dict[method])

        return

    def process_exps(self, exp_func):
        """
        该模块主要是设置模型解释的样本，分为单实例和多个实例。
        instance：单实例 定性解决问题时使用。
        test_set：多实例 定量分析效果。
        :param :exp_func 所选取的可解释方法。
        """

        if self.arg.data_level == 'instance':
            label_name_path = './resource/ntu_skeleton/label_name.txt'
            with open(label_name_path) as f:
                label_name = f.readlines()
                label_name = [line.rstrip() for line in label_name]
                self.label_name = label_name
            data_numpy, data, x, label, probability, skeleton_name = self.lead_data()

            output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_name)
            if not os.path.exists('{}/cam/'.format(output_result_dir)):
                os.makedirs('{}/cam/'.format(output_result_dir))

            frame_mask = self._set_frame_mask_(data, label)
            node_feature_mask = exp_func(frame_mask, data, x, label)

            if self.arg.plot_action:
                C, T, V, M = data_numpy.shape
                node_mask = data_reshape(data, node_feature_mask)
                wise = torch.sum(frame_mask * node_mask, dim=1)
                # wise = wise.unsqueeze(0)
                utils.visualization_skeleton.plot_action(data_numpy, self.model.graph.edge, wise.numpy(),
                                                         save_dir=output_result_dir,
                                                         save_type=self.arg.exp_type)

            return self.fidelity(frame_mask, node_feature_mask, data, label, probability)

        if self.arg.data_level == 'test_set':

            count = 0
            only_fidelity_acc_out = 0
            all_fidelity_acc_out = 0
            only_infidelity_acc_out = 0
            all_infidelity_acc_out = 0
            only_fidelity_prob_out = 0
            all_fidelity_prob_out = 0
            only_infidelity_prob_out = 0
            all_infidelity_prob_out = 0
            Sparsity_out = 0
            Sparsity_only = 0
            skeleton_file_path = './data/NTU-RGB-D/nturgb+d_skeletons/'
            # skeleton_file_path = './data/NTU-RGB-D/ntuall/'
            # skeleton_file_path = '/data/home/st/GT_CAM/st-gcn/data/NTU-RGB-D/nturgb+d_skeletons/'
            # skeleton_file_path = '/home/mxh/work_three/st-gcn/data/NTU-RGB-D/nturgb+d_skeletons/'
            for root, dirs, files in os.walk(skeleton_file_path):
                for file in files:
                    print("=================================================================")
                    print(file)
                    file_txt.write("=================================================================" + '\n')
                    file_txt.write(str(datetime.datetime.now()) + '\n')
                    skeleton_file = skeleton_file_path + file
                    data_numpy = read_xyz(
                        skeleton_file, max_body=max_body, num_joint=num_joint)
                    self.model.eval().to(self.dev)
                    data = torch.from_numpy(data_numpy)
                    data = data.unsqueeze(0)
                    data_model = data.float().to(self.dev)
                    with torch.no_grad():
                        out = self.model(data_model)
                        label = out.argmax(dim=1, keepdim=True).item()
                        probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
                    N, C, T, V, M = data.size()
                    x = data.permute(0, 2, 3, 4, 1).contiguous()
                    x = x.view(N * T * V * M, C)

                    output_result_dir = '{}/{}'.format(self.arg.output_dir, file)
                    if not os.path.exists('{}/cam/'.format(output_result_dir)):
                        os.makedirs('{}/cam/'.format(output_result_dir))

                    frame_mask = self._set_frame_mask_(data, label)
                    torch.set_printoptions(threshold=torch.inf, edgeitems=1000)
                    node_feature_mask = exp_func(frame_mask, data, x, label)
                    
                    if self.arg.plot_action:
                        node_mask = data_reshape(data, node_feature_mask)
                        wise = torch.sum(frame_mask * node_mask, dim=1)

                        utils.visualization_skeleton.plot_action(data_numpy, self.model.graph.edge, wise.numpy(),
                                                                 save_dir=output_result_dir,
                                                                 save_type=self.arg.exp_type)

                    only_fidelity_acc, all_fidelity_acc, only_infidelity_acc, all_infidelity_acc, \
                    only_fidelity_prob, all_fidelity_prob, only_infidelity_prob, all_infidelity_prob, Sparsity, Sparsityonly = \
                        self.evaluate(frame_mask, node_feature_mask, data, label, probability)
                    only_fidelity_acc_out += only_fidelity_acc
                    all_fidelity_acc_out += all_fidelity_acc
                    only_infidelity_acc_out += only_infidelity_acc
                    all_infidelity_acc_out += all_infidelity_acc
                    only_fidelity_prob_out += only_fidelity_prob
                    all_fidelity_prob_out += all_fidelity_prob
                    only_infidelity_prob_out += only_infidelity_prob
                    all_infidelity_prob_out += all_infidelity_prob
                    Sparsity_out += Sparsity
                    Sparsity_only += Sparsityonly
                    count = count + 1
                    print(count)
                    if count >= 2000:
                        break

                only_fidelity_acc = only_fidelity_acc_out / count
                all_fidelity_acc = all_fidelity_acc_out / count
                only_infidelity_acc = only_infidelity_acc_out / count
                all_infidelity_acc = all_infidelity_acc_out / count
                only_fidelity_prob = only_fidelity_prob_out / count
                all_fidelity_prob = all_fidelity_prob_out / count
                only_infidelity_prob = only_infidelity_prob_out / count
                all_infidelity_prob = all_infidelity_prob_out / count
                Sparsity = Sparsity_out / count
                Sparsityonly = Sparsity_only / count
                print("=================================================================")
                file_txt.write("=================================================================" + '\n')
                print("评价指标：指标忠诚度")
                file_txt.write("评价指标：指标忠诚度" + '\n')
                print("only_fidelity_acc:", format(only_fidelity_acc, '.3f'))
                file_txt.write("only_fidelity_acc:" + str(format(only_fidelity_acc, '.3f')) + '\n')
                print("all_fidelity_acc:", format(all_fidelity_acc, '.3f'))
                file_txt.write("all_fidelity_acc:" + str(format(all_fidelity_acc, '.3f')) + '\n')
                print("only_infidelity_acc:", format(only_infidelity_acc, '.3f'))
                file_txt.write("only_infidelity_acc:" + str(format(only_infidelity_acc, '.3f')) + '\n')
                print("all_infidelity_acc:", format(all_infidelity_acc, '.3f'))
                file_txt.write("all_infidelity_acc:" + str(format(all_infidelity_acc, '.3f')) + '\n')
                print("only_fidelity_prob:", format(only_fidelity_prob, '.3f'))
                file_txt.write("only_fidelity_prob:" + str(format(only_fidelity_prob, '.3f')) + '\n')
                print("all_fidelity_prob:", format(all_fidelity_prob, '.3f'))
                file_txt.write("all_fidelity_prob:" + str(format(all_fidelity_prob, '.3f')) + '\n')
                print("only_infidelity_prob:", format(only_infidelity_prob, '.3f'))
                file_txt.write("only_infidelity_prob:" + str(format(only_infidelity_prob, '.3f')) + '\n')
                print("all_infidelity_prob:", format(all_infidelity_prob, '.3f'))
                file_txt.write("all_infidelity_prob:" + str(format(all_infidelity_prob, '.3f')) + '\n')
                print("Sparsity:", format(Sparsity, '.3f'))
                file_txt.write("Sparsity:" + str(format(Sparsity, '.3f')) + '\n')
                print("Sparsityonly:", format(Sparsityonly, '.3f'))
                file_txt.write("Sparsityonly:" + str(format(Sparsityonly, '.3f')) + '\n')
                print("method:", format(self.arg.exp_type.lower()))
                print(self.arg.valid)
                print(self.arg.m)
                print(self.arg.w)
                file_txt.write("self.arg.valid:" + str(format(self.arg.valid)) + '\n')
                file_txt.write("method:" + str(format((self.arg.exp_type.lower())) + '\n'))
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
        # skeleton_file = './data/NTU-RGB-D/nturgb+d_skeletons/'
        skeleton_file = './data/NTU-RGB-D/ntuall/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = read_xyz(
            skeleton_file, max_body=max_body, num_joint=num_joint)
        self.model.eval().to(self.dev)
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

    def _set_frame_mask_(self, data, label):
        torch.manual_seed(2025)
        torch.cuda.manual_seed(2025)
        np.random.seed(2025)
        random.seed(2025)
        N, C, T, V, M = data.size()
        print("该动作的时间序列长度为：", T, "帧")
        file_txt.write("该动作的时间序列长度为：" + str(T) + "帧" + '\n')
        mask = torch.zeros_like(data)
        cishu = self.arg.m
        prob_box_random, starts, len_lenlenlen = self.frame_prob_random(data, label, cishu)
        n = self.arg.num
        # 将prob_box转为numpy数组，获取前n个最大值的索引
        prob_box_array = np.array(prob_box_random)
        top_n_indices = np.argsort(prob_box_array)[-n:]  # 获取最大n个值的索引

        # 根据这些索引从starts中获取对应位置的值
        top_n_positions = [starts[ttt] for ttt in top_n_indices]
        top_n_positions_end = [starts[ttt] + len_lenlenlen for ttt in top_n_indices]
        # print(top_n_positions)
        # print(top_n_positions_end)

        # 确保 starts 和 ends 有相同的长度
        assert len(top_n_positions) == len(top_n_positions_end), "starts 和 ends 列表长度必须相同"

        # 初始化交集的开始和结束位置
        intersection_start = max(top_n_positions)
        intersection_end = min(top_n_positions_end)

        # 判断是否存在交集
        if intersection_start <= intersection_end:
            starts_fin = intersection_start
            ends_fin = intersection_end
        else:
            # 如果没有交集，返回最后一组开始和结束位置
            starts_fin = top_n_positions[-1]
            ends_fin = top_n_positions_end[-1]            
        
        mask[:, :, starts_fin:ends_fin, :, :] = 1
        frame_mask = mask
        return frame_mask

    def frame_prob(self, data, label, T, pacing, stride):
        prob_box = []
        mask = torch.zeros_like(data)
        for i in range(0, T - pacing, stride):
            cover_mask = torch.zeros_like(data)
            cover_mask[:, :, i:i + pacing, :, :] = 1
            x_data = data * cover_mask
            prob = self.calculation(x_data, label)
            prob_box.append(float(format(prob, '.5f')))
        return prob_box

    def frame_prob_random(self, data, label, cishu):
            N, C, T, V, M = data.size()
            prob_box = []
            starts = []
            w =  self.arg.w
            len_lenlenlen = int(T * w)

            for i in range(0, cishu):
                mask = torch.zeros_like(data)
                while(True):
                    start = self.generate_random_integer(0,T)
                    if(start + len_lenlenlen <= T):
                        break
                mask[:, :, start: start + len_lenlenlen, :, :] = 1
                x_data = data * mask
                prob = self.calculation(x_data, label)
                prob_box.append(float(format(prob, '.5f')))
                starts.append(start)
            return prob_box, starts, len_lenlenlen

    def generate_random_integer(self,min_value, max_value):

        # 确保min_value和max_value是Python标量
        min_val = int(min_value)
        max_val = int(max_value)

        # 计算均值和标准差
        mean = (max_value + min_value) / 2
        std_dev = (max_value - min_value) / 6  # 通常取范围的1/6

        # 生成随机数
        random_value = torch.normal(mean=mean, std=std_dev, size=(1,))
        random_integer = int(round(random_value.item()))  # 明确转换为Python整数

        # 使用纯Python比较操作
        if random_integer < min_val:
            return min_val
        if random_integer > max_val:
            return max_val
        return random_integer

        # random_integer = torch.round(random_value).item()

        # # 使用Python内置的min/max函数
        # return max(min_val, min(random_integer, max_val))

        # random_integer = max(min_value, min(random_integer, max_value))
        # return int(random_integer)

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
            out = self.model(data)
            probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
        return probability

    # def __loss__(self, masked_pred, original_pred, mask):
    #     """
    #     该模块为损失函数。
    #     :param : log_logits: 掩膜处理后的预测结果
    #     :param : pred_label: 骨架动作的标签
    #     """
    #     self.coeffs = coeffs
    #     eps = torch.tensor([1e-7], device=self.dev)
    #     size_loss = torch.sum(mask) * coeffs['mask_size']
    #     mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    #     mask_ent_loss = coeffs['mask_ent'] * torch.mean(mask_ent_reg)
    #     cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
    #     loss = cce_loss + size_loss  
    #     return loss

    def __loss__(self, masked_pred, original_pred, mask, alpha=0.25, gamma=2.0):
        """
        该模块为损失函数，采用焦点损失（Focal Loss）。
        :param masked_pred: 掩膜处理后的预测结果
        :param original_pred: 骨架动作的标签
        :param mask: 掩膜
        :param alpha: 类别权重系数（平衡类间差异）
        :param gamma: 聚焦参数，用于减小简单样本的贡献
        """
        self.coeffs = coeffs
        # 计算交叉熵损失
        cross_entropy_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred, reduction='none')
        # 计算预测的概率
        prob = torch.nn.functional.softmax(masked_pred, dim=-1)
        # 计算焦点损失
        focal_loss = alpha * (1 - prob) * gamma * cross_entropy_loss
        # focal_loss = alpha * (1 - prob) ** gamma * cross_entropy_loss

        # 计算掩膜大小的惩罚项
        size_loss = torch.sum(mask) * self.coeffs['mask_size']
        
        # 计算掩膜的熵正则项
        eps = torch.tensor([1e-7], device=self.dev)
        mask_ent_reg = -mask * torch.log(mask + eps) - (1 - mask) * torch.log(1 - mask + eps)
        mask_ent_loss = self.coeffs['mask_ent'] * torch.mean(mask_ent_reg)
        
        # 总损失
        loss = torch.mean(focal_loss) + size_loss
        
        return loss


    def forward(self, frame_mask, data, change_data, label):
        self.model.eval()
        self.clear_masks()
        self._set_frame_mask_(data, label)
        frame_mask, mask = self.explainer(frame_mask, data, change_data, label)
        self.clear_masks()
        return mask

    def explainer(self, frame_mask, data, change_data, label):
        N, C, T, V, M = frame_mask.size()
        x = frame_mask.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(N * T * V * M, C)
        change_data = change_data.to(self.dev)
        # data = data * frame_mask

        data = data.float().to(self.dev)
        self.model.eval()
        self.model.zero_grad()
        with torch.no_grad():
            self.model.eval()
            output = self.model(data)
            prediction_label = output.argmax(dim=1, keepdim=True)
        self._set_feature_masks_(change_data)
        parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        bast_output = torch.tensor([0]).to(self.dev)
        index = 0
        for epoch in range(1, self.epochs + 1):
            self.model.eval()
            mask = (self.node_feat_mask.sigmoid() * x).to(self.dev)
            h = change_data * mask
            h = data_reshape(data=data, mask=h).float()
            out = self.model(h)
            test_out = torch.nn.functional.softmax(out[0], dim=-1)
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
            
        bast_feat_mask = self.node_feat_mask.detach()
        bast_feat_mask = bast_feat_mask.sigmoid()
        node_mask = data_reshape(data, bast_feat_mask)
        # with torch.no_grad():
        #     self.model.eval()
        #     output = self.model((data*node_mask*frame_mask).float())
        #     prediction_label = output.argmax(dim=1, keepdim=True)
        #
        #     print(prediction_label)

        self.clear_masks()

        return bast_feat_mask

    def evaluate(self, frame_mask, node_feature_mask, data, label, probability):
        return self.fidelity(frame_mask, node_feature_mask, data, label, probability)

    def fidelity(self, frame_mask, node_feature_mask, data, label, probability):

        All_one_mask = torch.ones_like(data)
        node_mask = data_reshape(data, node_feature_mask)
        un_frame_mask = All_one_mask - frame_mask
        un_node_mask = All_one_mask - node_mask
        print("模型识别的类别：", label, "   类别概率：", format(probability, '.4f'))
        file_txt.write("模型识别的类别：" + str(label) + "   类别概率：" + str(format(probability, '.4f')) + '\n')
        # 计算fidelity+ acc 和 fidelity+ prob
        self.model.eval()
        with torch.no_grad():
            # 只使用帧掩码
            only_frame_in = (data * frame_mask).float().to(self.dev)
            only_frame_out = self.model(only_frame_in)
            # 记录只有帧特征的概率和标签
            only_frame_label = only_frame_out.argmax(dim=1, keepdim=True).item()
            only_frame_prob = torch.nn.functional.softmax(only_frame_out[0], dim=-1)[label].item()
            print("只使用帧特征掩码类别：", only_frame_label, "     只使用帧特征掩码概率：",
                  format(only_frame_prob, '.4f'))
            file_txt.write(
                "只使用帧特征掩码类别：" + str(only_frame_label) + "   只使用帧特征掩码概率：" + str(
                    format(only_frame_prob, '.4f')) + '\n')

            # 使用帧特征和节点特征
            All_in = (data * frame_mask * node_mask).float().to(self.dev)
            All_out = self.model(All_in)
            # 记录帧特征和节点特征的概率和标签
            All_label = All_out.argmax(dim=1, keepdim=True).item()
            All_prob = torch.nn.functional.softmax(All_out[0], dim=-1)[label].item()
            print("使用所有掩码类别：", All_label, "     使用所有掩码概率：", format(All_prob, '.4f'))
            file_txt.write(
                "使用所有掩码类别：" + str(All_label) + "   使用所有掩码概率：" + str(format(All_prob, '.4f')) + '\n')
            # 只使用帧补掩码
            only_un_frame_in = (data * un_frame_mask).float().to(self.dev)
            only_un_frame_out = self.model(only_un_frame_in)
            # 记录只有帧补掩码的概率和标签
            only_un_frame_label = only_un_frame_out.argmax(dim=1, keepdim=True).item()
            only_un_frame_prob = torch.nn.functional.softmax(only_un_frame_out[0], dim=-1)[label].item()
            print("只使用不重要帧特征掩码类别：", only_un_frame_label, "     只使用不重要帧特征掩码概率：",
                  format(only_un_frame_prob, '.4f'))
            file_txt.write(
                "只使用不重要帧特征掩码类别：" + str(only_un_frame_label) + "   只使用不重要帧特征掩码概率：" + str(
                    format(only_un_frame_prob, '.4f')) + '\n')
            # 使用帧特征和节点特征
            All_un_in = (data * un_frame_mask * un_node_mask).float().to(self.dev)
            All_un_out = self.model(All_un_in)
            # 记录帧特征和节点特征的概率和标签
            All_un_label = All_un_out.argmax(dim=1, keepdim=True).item()
            All_un_prob = torch.nn.functional.softmax(All_un_out[0], dim=-1)[label].item()
            print("使用所有不重要掩码类别：", All_un_label, "     使用所有不重要掩码概率：", format(All_un_prob, '.4f'))
            file_txt.write("使用所有不重要掩码类别：" + str(All_un_label) + "   使用所有不重要掩码概率：" + str(
                format(All_un_prob, '.4f')) + '\n')

        # 开始计算Fidelity_ACC
        acc_frame = 1 if label == only_frame_label else 0
        acc_all = 1 if label == All_label else 0
        acc_un_frame = 1 if label == only_un_frame_label else 0
        acc_un_all = 1 if label == All_un_label else 0
        # 开始计算prob
        prob_frame = probability - only_frame_prob
        prob_all = probability - All_prob
        prob_un_frame = probability - only_un_frame_prob
        prob_un_all = probability - All_un_prob

        # 只有不重要帧mask的正确率
        only_fidelity_acc = 1 - acc_un_frame
        print("only_fidelity_acc:", only_fidelity_acc)
        file_txt.write("only_fidelity_acc:" + str(only_fidelity_acc) + '\n')
        # 所有不重要帧mask和节点特征的mask的正确率
        all_fidelity_acc = 1 - acc_un_all
        print("all_fidelity_acc:", all_fidelity_acc)
        file_txt.write("all_fidelity_acc:" + str(all_fidelity_acc) + '\n')
        # 只有重要帧mask的正确率
        only_infidelity_acc = 1 - acc_frame
        print("only_infidelity_acc:", only_infidelity_acc)
        file_txt.write("only_infidelity_acc:" + str(only_infidelity_acc) + '\n')
        # 所有重要帧mask和节点特征的mask的正确率
        all_infidelity_acc = 1 - acc_all
        print("all_infidelity_acc:", all_infidelity_acc)
        file_txt.write("all_infidelity_acc:" + str(all_infidelity_acc) + '\n')

        # 只有不重要帧mask的概率
        only_fidelity_prob = prob_un_frame
        print("only_fidelity_prob:", format(only_fidelity_prob, '.4f'))
        file_txt.write("only_fidelity_prob:" + str(only_fidelity_prob) + '\n')
        # 所有不重要帧mask和节点特征的mask的正确率
        all_fidelity_prob = prob_un_all
        print("all_fidelity_prob:", format(all_fidelity_prob, '.4f'))
        file_txt.write("all_fidelity_prob:" + str(all_fidelity_prob) + '\n')
        # 只有重要帧mask的概率
        only_infidelity_prob = prob_frame
        print("only_infidelity_prob:", format(only_infidelity_prob, '.4f'))
        file_txt.write("only_infidelity_prob:" + str(only_infidelity_prob) + '\n')
        # 所有重要帧mask和节点特征的mask的正确率
        all_infidelity_prob = prob_all
        print("all_infidelity_prob:", format(all_infidelity_prob, '.4f'))
        file_txt.write("all_infidelity_prob:" + str(all_infidelity_prob) + '\n')
        # 计算Sparsity
        Sparsity =1- torch.sum(frame_mask * node_mask) / torch.sum(All_one_mask).item()
        print("Sparsity：", format(Sparsity, '.4f'))
        file_txt.write("Sparsity：" + str(format(Sparsity, '.4f')) + '\n')
        Sparsityonly =1- torch.sum(frame_mask) / torch.sum(All_one_mask).item()
        print("Sparsityonly：", format(Sparsityonly, '.4f'))
        file_txt.write("Sparsityonly：" + str(format(Sparsityonly, '.4f')) + '\n')

        return only_fidelity_acc, all_fidelity_acc, only_infidelity_acc, all_infidelity_acc, \
               only_fidelity_prob, all_fidelity_prob, only_infidelity_prob, all_infidelity_prob, Sparsity, Sparsityonly

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
                            default = True,
                            help='save action as image',
                            type=bool)
        parser.add_argument('--output_dir',
                            default='./work3result_v/',
                            help='Path to save results')
        parser.add_argument('--exp_type',
                            default='explainer',
                            help='one of gnn_explainer,pg_explainer,GNN_Explainer')
        parser.add_argument('--data_level',
                            default='test_set',
                            help='instance or test_set')
        parser.add_argument('--valid',
                            default='csetup',
                            help='One of xsub and xview,csetup,csub')
        parser.add_argument('--num',
                             type=int, default=8)
        parser.add_argument('--m',
                             type=int, default=50)
        parser.add_argument('--w',
                             type=float, default=0.5)
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
        #     config='./config/st_gcn/ntu-{}/test.yaml'.format(parent_parser.valid))
        ntu60_valids = ['xsub', 'xview']
        valid = args[0].valid
        if valid in ntu60_valids:
            config_path = './config/st_gcn/ntu-{}/test.yaml'.format(parent_parser.valid)  # cff,gnnexplainer
        else:
            config_path = './config/st_gcn/ntu120-{}/test.yaml'.format(parent_parser.valid)
        parser.set_defaults(config=config_path)
        parser.set_defaults(print_log=False)
        print(f"Using config: {config_path}")
        # parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
