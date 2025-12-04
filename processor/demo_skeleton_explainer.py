import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import time
import os
import numpy as np
from .processor import Processor
from math import sqrt
from torch.nn.functional import cross_entropy
from tools.utils.ntu_read_skeleton import read_xyz,read_xyz_new
from torch_geometric.nn import MessagePassing




max_body = 2
num_joint = 25
max_frame = 300

drop_stride = 5
EPS = 1e-15
coeffs = {
    'edge_size': 0.005,
    'edge_reduction': 'sum',
    'node_feat_size': 1.0,
    'node_feat_reduction': 'mean',
    'edge_ent': 1.0,
    'node_feat_ent': 0.1,
}

# 记录结果
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
# file_txt = open('result/'+time_now+'.txt', mode='a', encoding="utf-8")
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)  # 确保目录存在
file_txt = open(os.path.join(result_dir, time_now+'.txt'), mode='a', encoding="utf-8")

# 交叉熵损失函数
def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)

class DomeSkeletonExplainer(Processor):

    def start(self):
        print("Using {} weights.".format(self.arg.valid))
        exp_dict = {'gnn_explainer': self.GNN_Explainer,
                    'pg_explainer': self.PG_Explainer}
        mothod = self.arg.exp_type.lower()
        out = self.process_exps(exp_dict[mothod])

        return

    def process_exps(self, exp_func):

        if self.arg.data_level == 'instance':
            label_name_path = './resource/ntu_skeleton/label_name.txt'
            with open(label_name_path) as f:
                label_name = f.readlines()
                label_name = [line.rstrip() for line in label_name]
                self.label_name = label_name
            data, x, label = self.lead_data()
            edge_index = self.get_edge_index()
            node_feat_mask, pre_label = exp_func(data, x, edge_index, label)

            return self.evaluate(data=data, x=x, node_feature_mask=node_feat_mask, label=pre_label)

        elif self.arg.data_level == 'test_set':
            loader = self.data_loader['test']
            count = 0
            fidelity_acc_out = 0
            fidelity_prob_out = 0
            infidelity_acc_out = 0
            infidelity_prob_out = 0

            file_txt.write("TEST_SET:"+'\n')

            for data, label in loader:
                print("=================================================================")
                file_txt.write("================================================================="+'\n')
                file_txt.write(str(datetime.datetime.now())+'\n')
                str_label = label[0].item()
                print("Skeleton class:", str_label)
                file_txt.write("Skeleton class:"+str(str_label)+'\n')
                x = self.change_data_to_mask(data)
                edge_index = self.get_edge_index()
                node_feature_mask, pre_label = exp_func(data, x, edge_index, label)
                fidelity_acc, fidelity_prob,infidelity_acc, infidelity_prob = self.evaluate(x=x, data=data, node_feature_mask=node_feature_mask, label=pre_label)
                fidelity_acc_out += fidelity_acc
                fidelity_prob_out += fidelity_prob
                infidelity_acc_out += infidelity_acc
                infidelity_prob_out += infidelity_prob
                count = count + 1
                if count >= 60:
                    break

            fidelity_acc = fidelity_acc_out / count
            fidelity_prob = fidelity_prob_out / count
            infidelity_acc = infidelity_acc_out / count
            infidelity_prob = infidelity_prob_out / count
            print("=================================================================")
            file_txt.write("================================================================="+'\n')
            print("评价指标：指标忠诚度")
            file_txt.write("评价指标：指标忠诚度"+'\n')
            print("fidelity_acc:", format(fidelity_acc, '.3f'))
            file_txt.write("fidelity_acc:"+str(format(fidelity_acc, '.3f'))+'\n')
            print("fidelity_prob:", format(fidelity_prob, '.3f'))
            file_txt.write("fidelity_prob:"+str(format(fidelity_prob, '.3f'))+'\n')
            print("infidelity_acc:", format(infidelity_acc, '.3f'))
            file_txt.write("infidelity_acc:"+str(format(infidelity_acc, '.3f'))+'\n')
            print("infidelity_prob:", format(infidelity_prob, '.3f'))
            file_txt.write("infidelity_prob:"+str(format(infidelity_prob, '.3f'))+'\n')
            file_txt.close()

        elif self.arg.data_level == 'test':

            count = 0
            fidelity_acc_out = 0
            fidelity_prob_out = 0
            infidelity_acc_out = 0
            infidelity_prob_out = 0

            skeleton_file_path = 'D:\\s\\st-gcn\\data\\NTU-RGB-D\\nturgb+d_skeletons\\'
            for root, dirs, files in os.walk(skeleton_file_path):
                for file in files:
                    print("=================================================================")
                    file_txt.write("=================================================================" + '\n')
                    file_txt.write(str(datetime.datetime.now()) + '\n')
                    skeleton_file = skeleton_file_path + file
                    data_numpy = read_xyz(
                        skeleton_file, max_body=max_body, num_joint=num_joint)
                    self.model.eval()
                    data = torch.from_numpy(data_numpy)
                    data = data.unsqueeze(0)
                    data_model = data.float().to(self.dev)
                    with torch.no_grad():
                        out = self.model(data_model)
                        label = out.argmax(dim=1, keepdim=True).item()
                    N, C, T, V, M = data.size()
                    x = data.permute(0, 2, 3, 4, 1).contiguous()
                    x = x.view(N * T * V * M, C)
                    x = self.change_data_to_mask(data)
                    edge_index = self.get_edge_index()
                    node_feature_mask, pre_label = exp_func(data, x, edge_index, label)
                    fidelity_acc, fidelity_prob, infidelity_acc, infidelity_prob = self.evaluate(x=x, data=data,
                                                                                                 node_feature_mask=node_feature_mask,
                                                                                                 label=pre_label)
                    fidelity_acc_out += fidelity_acc
                    fidelity_prob_out += fidelity_prob
                    infidelity_acc_out += infidelity_acc
                    infidelity_prob_out += infidelity_prob
                    count = count + 1
                    if count >= 60:
                        break

                fidelity_acc = fidelity_acc_out / count
                fidelity_prob = fidelity_prob_out / count
                infidelity_acc = infidelity_acc_out / count
                infidelity_prob = infidelity_prob_out / count
                print("=================================================================")
                file_txt.write("=================================================================" + '\n')
                print("评价指标：指标忠诚度")
                file_txt.write("评价指标：指标忠诚度" + '\n')
                print("fidelity_acc:", format(fidelity_acc, '.3f'))
                file_txt.write("fidelity_acc:" + str(format(fidelity_acc, '.3f')) + '\n')
                print("fidelity_prob:", format(fidelity_prob, '.3f'))
                file_txt.write("fidelity_prob:" + str(format(fidelity_prob, '.3f')) + '\n')
                print("infidelity_acc:", format(infidelity_acc, '.3f'))
                file_txt.write("infidelity_acc:" + str(format(infidelity_acc, '.3f')) + '\n')
                print("infidelity_prob:", format(infidelity_prob, '.3f'))
                file_txt.write("infidelity_prob:" + str(format(infidelity_prob, '.3f')) + '\n')
                file_txt.close()

    def get_edge_index(self):

        edge_index = torch.Tensor([[0,0,0,1,1,2,2,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,
                                         11,11,12,12,13,13,14,14,15,16,16,17,17,18,18,19,
                                         20,20,20,20,21,22,22,23,24,24],
                                        [1,12,16,0,20,3,20,2,5,20,4,6,5,7,6,22,9,20,8,10,
                                         9,11,10,24,0,13,12,14,13,15,14,0,17,16,18,17,19,
                                         18,1,2,4,8,22,7,21,24,11,23]])
        return edge_index

    def lead_data(self):
        skeleton_name = self.arg.skeleton
        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
        print("Processing skeleton:", skeleton_name)
        print("Skeleton class:",action_class)
        self.action_class = action_class

        out_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_name)

        skeleton_file = 'D:\\s\\st-gcn\\data\\NTU-RGB-D\\nturgb+d_skeletons\\'
        # skeleton_file = '../data/NTU-RGB-D/nturgb+d_skeletons/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = read_xyz(
            skeleton_file, max_body=max_body, num_joint=num_joint)

        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data_model = data.float().to(self.dev)
        with torch.no_grad():
            out = self.model(data_model)
            label = out.argmax(dim=1, keepdim = True).item()
        N,C,T,V,M = data.size()
        x = data.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(N*T*V*M, C)
        return data, x, label

    def change_data_to_mask(self, data):
        N,C,T,V,M = data.size()
        data1 = data.permute(0, 2, 3, 4, 1).contiguous()
        data1 = data1.view(N*T*V*M, C)

        x = data1

        return x

    def PG_Explainer(self):
        return

    def _set_masks_(self, x, edge_index, init = "normal"):
        (N,F), E = x.size(), edge_index.size(1)

        self.allow_edge_mask = False

        std = 0.1
        if self.arg.feat_mask_type == 'individual_feature':
            # 固定mask
            # self.node_feat_mask = torch.nn.Parameter(torch.ones_like(x) * std)
            # 随机mask
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)
        elif self.arg.feat_mask_type == 'cam':
            self.node_feat_mask = torch.nn.Parameter(x)
        elif self.arg.feat_mask_type == 'scalar':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(50, 1) * std)
        else:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        if not self.allow_edge_mask:
            self.edge_mask.requires_grad_(False)
            self.edge_mask.fill_(float('inf'))  # `sigmoid()` returns `1`.
        self.loop_mask = edge_index[0] != edge_index[1]

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
                module.__loop_mask__ = self.loop_mask

    def clear_masks(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
                module.__loop_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None
        module.loop_mask = None

    def __loss__(self, node_idx, log_logits, pred_label, mask):
        self.return_type = 'raw'
        self.coeffs = coeffs

        if self.return_type == 'regression':
            if node_idx != -1:
                loss = torch.cdist(log_logits[node_idx], pred_label[node_idx])
            else:
                loss = torch.cdist(log_logits, pred_label)
        else:
            if node_idx != -1:
                loss = -log_logits[node_idx, pred_label[node_idx]]
            else:
                loss = -log_logits[0, pred_label[0]]

        # 增加一个loss约束项 mask的离散程度
        # loss = loss - 0.1*mask.std()
        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def GNN_Explainer(self, data, x, edge_index, label, **kwargs):
        self.epochs = 100

        self.model.eval()
        self.model.zero_grad()
        self.clear_masks()
        N,C,T,V,M = data.size()
        data = data.float().to(self.dev).requires_grad_()

        # all nodes belong to same graph
        # Get the initial prediction. 初次进行识别
        with torch.no_grad():
            output = self.model(data)
            pre_out = torch.nn.functional.softmax(output[0],dim=-1)
            prediction_label = output.argmax(dim=1, keepdim = True)
            print("真实类别的概率：", format(pre_out[label].item(), '.3f'))
            # file_txt.write("真实类别的概率："+str(format(pre_out[label].item(), '.3f'))+'\n')
            print("预测类别的概率：", format(pre_out[prediction_label][0].item(), '.3f'))
            # file_txt.write("预测类别的概率："+str(format(pre_out[prediction_label][0].item(), '.3f'))+'\n')

            # 设置初始掩膜
        if self.arg.mask_type == 'cam':
            self.cfg_hook()
            print("开始使用Cam为初始掩膜")
            file_txt.write("开始使用Cam为初始掩膜" + '\n')
            s_map = self.GradCam(data)
            s_map = self.change_data_to_mask(s_map)
            s_map = self.mask_cat(data, s_map)
            self._set_masks_(s_map, edge_index)
        elif self.arg.mask_type == 'random':
            self._set_masks_(x, edge_index)


        #  进行掩膜的学习和优化

        if self.allow_edge_mask:
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=0.01)
        # 设置早停法
        bast_output = torch.tensor([0])
        index = 1

        for epoch in range(1, self.epochs+1):
            optimizer.zero_grad()
            mask = self.node_feat_mask.sigmoid()
            h = x * mask
            h = self.data_reshape(data=data, mask=h)
            h = h.float().to(self.dev).requires_grad_()
            out = self.model(h)
            test_out = torch.nn.functional.softmax(out[0], dim=-1)
            if test_out[prediction_label] > bast_output:
                bast_output = test_out[prediction_label]
                index = epoch
            else:
                if epoch - index > 5:
                    break
            loss = self.__loss__(-1, torch.log(test_out).unsqueeze(0), pred_label=prediction_label, mask=self.node_feat_mask)
            loss.backward()
            optimizer.step()
        bast_feat_mask = self.node_feat_mask.detach().sigmoid()
        self.clear_masks()

        return bast_feat_mask, prediction_label

    # def visualization(self,node_feat_mask):
    #     mpl.rcParams['font.size'] = 10
    #     fig = plt.figure(figsize=(10,10))
    #     ax = fig.add_subplot(111, projection='3d')
    #     N,F = node_feat_mask.size()
    #     for n in range(N):
    #         x = node_feat_mask[n,0]
    #         y = node_feat_mask[n,1]
    #         z = node_feat_mask[n,2]
    #         ax.scatter(x,y,z,cmap='viridis',linewidth=1)
    #
    #     plt.show()

    def mask_cat(self, data, mask):
        N,C,T,V,M = data.size()
        # mask.shape = (V*M,C)
        mask_type = 'individual_feature'
        if mask_type == 'individual_feature':
            return mask.repeat(1, C)
        elif mask_type == 'scalar':
            return mask


    # 评价指标
    def evaluate(self,data,x,node_feature_mask,label):

        return self.fidelity(data, x, node_feature_mask, label)

    def fidelity(self,data,x,node_feature_mask, label):
        # 计算fidelity+ acc 和 fidelity+ prob
        self.model.eval()
        one_node_feat_mask = torch.ones_like(x)
        unimportance_node_feat_mask = one_node_feat_mask - node_feature_mask

        with torch.no_grad():
            # 真实的输出和类别
            data = data.float().to(self.dev)
            output = self.model(data)
            # file_txt.write("原始输出结果："+str(output))
            out_softmax = torch.nn.functional.softmax(output[0],dim=-1)
            out_label = output.argmax(dim=1, keepdim = True).item()
            prediction_prob = out_softmax[label].item()
            print("模型识别的类别：", out_label,"   类别概率：", format(prediction_prob, '.4f'))
            file_txt.write("模型识别的类别："+str(out_label)+"   类别概率：" + str(format(prediction_prob, '.4f')) + '\n')
            # 重要特征的输出和类别
            node_feature_mask = x * node_feature_mask
            node_feature_mask = self.data_reshape(data, node_feature_mask)
            node_feature = node_feature_mask.float().to(self.dev)
            feature_out = self.model(node_feature)
            feature_out_softmax = torch.nn.functional.softmax(feature_out[0],dim=-1)
            feature_label = feature_out.argmax(dim=1, keepdim = True).item()
            feature_prob = feature_out_softmax[out_label].item()
            print("使用特征掩码类别：", feature_label,"   使用特征掩码概率：", format(feature_prob, '.4f'))
            file_txt.write("使用特征掩码类别："+str(feature_label)+"   使用特征掩码概率：" + str(format(feature_prob, '.4f')) + '\n')
            # 非重要特征的输出和类别
            unimportance_node_feat_mask = x * unimportance_node_feat_mask
            unimportance_node_feat_mask = self.data_reshape(data, unimportance_node_feat_mask)
            unimportance_node_feature = unimportance_node_feat_mask.float().to(self.dev)
            unfeature_output = self.model(unimportance_node_feature)
            unfeature_output_softmax = torch.nn.functional.softmax(unfeature_output[0],dim=-1)
            unfeature_label = unfeature_output.argmax(dim=1, keepdim = True).item()
            unfeature_prob = unfeature_output_softmax[out_label].item()
            print("使用非特征掩码类别：", unfeature_label,"   使用非特征掩码概率：", format(unfeature_prob, '.4f'))
            file_txt.write("使用非特征掩码类别："+str(unfeature_label)+"   使用非特征掩码概率：" + str(format(unfeature_prob, '.4f')) + '\n')

            # 计算fidelity+ acc
            if label == out_label:
                index_x = 1
            else:
                index_x = 0
            if label == unfeature_label:
                index_y = 1
            else:
                index_y = 0
            if label == feature_label:
                index_z = 1
            else:
                index_z = 0
            fidelity_acc = index_x-index_y
            print("fidelity_acc:", fidelity_acc)
            file_txt.write("fidelity_acc:"+str(fidelity_acc) + '\n')
            # 计算fidelity+ prob
            fidelity_prob = prediction_prob - unfeature_prob
            print("fidelity_prob:", format(fidelity_prob, '.4f'))
            file_txt.write("fidelity_prob:"+str(fidelity_prob) + '\n')
            # 计算infidelity_acc
            infidelity_acc = index_x-index_z
            print("infidelity_acc:", infidelity_acc)
            file_txt.write("infidelity_acc:"+str(infidelity_acc) + '\n')
            # 计算infidelity——prob
            infidelity_prob = prediction_prob - feature_prob
            print("infidelity_prob:", format(infidelity_prob,'.4f'))
            file_txt.write("infidelity_prob:"+str(infidelity_prob) + '\n')
            return  fidelity_acc, fidelity_prob,infidelity_acc,infidelity_prob

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

    def GradCam(self, data, action_class=None):
        '''GradCAM 方法'''
        if self.arg.data_level == "test_set":
            assert self.arg.test_batch_size == 1, "Test batch size for GradCAM method must be one."
        # 预处理，获取激活和梯度
        # inference
        self.model.eval()
        self.model.zero_grad()

        # action recognition
        data = data.to(torch.float32)

        model_output = self.model(data)

        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)

        self.loss = torch.nn.CrossEntropyLoss()
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

        return saliency_map

    def reshape_saliency_map(self, saliency_map, data):
        bs, c, t, v, m = saliency_map.shape
        saliency_map = torch.nn.functional.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        saliency_map = self.data_norm(saliency_map)
        return saliency_map

    def data_norm(self, data):
        bs, c, t, v, m = data.shape
        data_min = data.view(bs, -1).min(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
        data_max = data.view(bs, -1).max(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
        # denominator = torch.where(
        #     (data_max-data_min) != 0., data_max-data_min, torch.tensor(1.).double().to(self.dev))
        denominator = torch.where(
            (data_max-data_min) != 0., data_max-data_min, torch.tensor(1.).to(self.dev))

        return (data - data_min) / denominator

    def discreat_mask(self, mask):
        t,c = mask.size()
        x_means = mask[:,0].mean()
        y_means = mask[:,1].mean()
        z_means = mask[:,2].mean()
        means_mask = torch.ones_like(mask)
        means_mask[:,0]=x_means
        means_mask[:,1]=y_means
        means_mask[:,2]=z_means
        mean_mask = torch.ones_like(mask)
        for i in range(0,t,25):
            x_mean = mask[i:i+25,0].mean()
            y_mean = mask[i:i+25,1].mean()
            z_mean = mask[i:i+25,2].mean()
            mean_mask[i:i+25,0] = x_mean
            mean_mask[i:i+25,1] = y_mean
            mean_mask[i:i+25,2] = z_mean
        discreat_mask = (mask>=means_mask).float()
        for i in range(0,t,25):
            if discreat_mask[i:i+25,:].mean() < 35/75:
                discreat_mask[i:i+25,:] = 0
            else:
                discreat_mask[i:i+25,:] = 1
        # for i in range(0,t):
        #     if discreat_mask[i:i+1,:].mean() < 1/3:
        #         discreat_mask[i:i+1,:] = 0
        #     else:
        #         discreat_mask[i:i+1,:] = 1
        #
        # for i in range(0, t ,25):
        #     print(discreat_mask[i:i+25,:])
        return discreat_mask

    def data_reshape(self, data, mask):
        N,C,T,V,M = data.size()
        reshape_data = torch.ones_like(data)
        for i in range(T):
            for j in range(V):
                reshape_data[:,0,i,j,:] = mask[i*25+j,0]
                reshape_data[:,1,i,j,:] = mask[i*25+j,1]
                reshape_data[:,2,i,j,:] = mask[i*25+j,2]
        return reshape_data

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help = False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Dome for ST-GCN Explainer')
        parser.add_argument('--skeleton',
                            default='S001C001P001R001A005',
                            help='Path to video')
        parser.add_argument('--output_dir',
                            default='./data/demo_skeleton/result',
                            help='Path to save results')
        parser.add_argument('--exp_type',
                            default='gnn_explainer',
                            help='one of gnn_explainer,pg_explainer')
        parser.add_argument('--data_level',
                            default='test_set',
                            help='instance or test_set')
        parser.add_argument('--valid',
                            default='xsub',
                            help='One of xsub and xview')
        parser.add_argument('--mask_type',
                            default='random',
                            help='one of cam or random')
        parser.add_argument('--feat_mask_type',
                            default='individual_feature',
                            help='individual_feature or cam')
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

        args = parser.parse_known_args(namespace=parent_parser)
        # print("mark", parent_parser.valid)
        parser.set_defaults(
            config='./config/st_gcn/ntu-{}/test.yaml'.format(parent_parser.valid))
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser


