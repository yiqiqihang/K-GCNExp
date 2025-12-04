import copy
import itertools
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
# from geopy.distance import geodesic
# from geopy.point import Point
from net.st_gcn import Model
# import torchsnooper
import torch.nn.functional as F
import datetime
import time
import gc
from itertools import combinations, chain
# from stgcn_lrp.processor.skeleton_lrp import SkeletonLRP

# 加载预训练模型的特征提取部分
model_args = {'in_channels': 5, 'num_class': 10, 'dropout': 0, 'edge_importance_weighting': False,
              'graph_args': {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}} 


# pre_model_path =  r"models\STGCN_model.pt"
pre_model_path = './models/STGCN_model.pt'
pretrained_model = Model(**(model_args))
pretrained_model.load_state_dict(torch.load(pre_model_path))
pretrained_model.eval()



def load_model(all_data,label,index):

    # print(pretrained_model)
    accurate = 0
    total = 0
    # for index in range(len(all_data["Longitude1"])):
    #     if(label[index]!=4):
    #         continue
    long1 = all_data["Longitude1"][index].copy()
    long2 = all_data["Longitude2"][index].copy()
    la1 = all_data["Latitude1"][index].copy()
    la2 = all_data["Latitude2"][index].copy()
    a1 = all_data["Altitude1"][index].copy()
    a2 = all_data["Altitude2"][index].copy()

    tar0 = all_data["ReadltarRng0"][index].copy()
    tar1 = all_data["ReadltarRng1"][index].copy()
    tar2 = all_data["ReadltarRng2"][index].copy()
    tar3 = all_data["ReadltarRng3"][index].copy()
    data_val = np.zeros((1,5, len(long1), 2, 1))
    for nums in range(len(long1)):
        data_val[0,:,nums,0,0] = [long1[nums],la1[nums],a1[nums],tar0[nums],tar1[nums]]
        data_val[0,:,nums,1,0] = [long2[nums],la2[nums],a2[nums],tar2[nums],tar3[nums]]
        # data_numpy = np.array(data_val)
    data_numpy = data_val.astype(np.float)
    data_numpy = torch.Tensor(data_numpy)
        # label = label.long()
    output = pretrained_model(data_numpy)
    output = torch.softmax(output, dim=1)
    percentage_top1, predicted = torch.max(output.data, dim=1)

    percentage_top2 = np.sort(output.data.cpu().numpy())
    percentage_top2 = percentage_top2[0][-2]
    top2_Index = np.argsort(output.data.cpu().numpy())
    top2_Index = top2_Index[0][-2]
    if(predicted[0].numpy() == label[index]):
        # print("yes")
        return True
    else:
        # print("no")
        return False
        #     accurate = accurate+1
        # total = total+1
        # print("top1: " ,percentage_top1.numpy(),predicted.numpy() ,"top2: " ,percentage_top2,top2_Index,len(long1))
    # print(total)
    # print(accurate)
    # print(accurate/total)

unique_keys = ['p1', 'p2']
def load_test(all_data,label):
    accurate = 0
    total = 0
    ave_drop_all = 0
    ave_increase_all = 0
    inse_auc_all = 0
    del_auc_all = 0
    count = 0
    ave_drop = 0
    increase = 0
    inse_auc = 0
    del_auc = 0
    # 记录结果
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    # # file_txt = open('result-forcam-20240304/' + "ntu60-xsub" + time_now + '.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/crcam-{time_now}.txt', mode='a', encoding="utf-8")
    file_txt = open(f'result/gradcam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/bicam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/ablation-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/scorecam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/gradcampp-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'/data/home/st/GT_CAM/st-gcn/test/result/shapley-{time_now}.txt', mode='a', encoding="utf-8")
    for index in range(len(all_data["Longitude1"])):


        print(index)
        # if(all_data["Type"][index]!= 3 ):
        #     continue
        # 以下注释代码，通过index可以单测某个样本情况
        # if (index!=2):
        #     continue
        pretrained_model.eval()
        file_txt.write("=================================================================" + '\n')
        file_txt.write(str(datetime.datetime.now()) + '\n')
        long1 = all_data["Longitude1"][index].copy()
        long2 = all_data["Longitude2"][index].copy()
        la1 = all_data["Latitude1"][index].copy()
        la2 = all_data["Latitude2"][index].copy()
        a1 = all_data["Altitude1"][index].copy()
        a2 = all_data["Altitude2"][index].copy()

        tar0 = all_data["ReadltarRng0"][index].copy()
        tar1 = all_data["ReadltarRng1"][index].copy()
        tar2 = all_data["ReadltarRng2"][index].copy()
        tar3 = all_data["ReadltarRng3"][index].copy()

        # start = 0
        # end = 250
        #
        # long1 = long1[start:end]
        # long2 = long2[start:end]
        # la1 = la1[start:end]
        # la2 = la2[start:end]
        # a1 = a1[start:end]
        # a2 = a2[start:end]
        # tar0 = tar0[start:end]
        # tar1 = tar1[start:end]
        # tar2 = tar2[start:end]
        # tar3 = tar3[start:end]
        data_val = np.zeros((1, 5, len(long1), 2, 1))
        for nums in range(min(len(long1),len(long2),len(tar1))):
            data_val[0, :, nums, 0, 0] = [long1[nums], la1[nums], a1[nums], tar0[nums], tar1[nums]]
            data_val[0, :, nums, 1, 0] = [long2[nums], la2[nums], a2[nums], tar2[nums], tar3[nums]]
            # data_numpy = np.array(data_val)
        data_numpy = data_val
        data_numpy = torch.Tensor(data_numpy).to(torch.float32)
        # label = label.long()
        output_org = pretrained_model(data_numpy)
        output = torch.softmax(output_org, dim=1)
        percentage_top1, predicted = torch.max(output.data, dim=1)

        percentage_top2 = np.sort(output.data.cpu().numpy())
        percentage_top2 = percentage_top2[0][-2]
        top2_Index = np.argsort(output.data.cpu().numpy())
        top2_Index = top2_Index[0][-2]
        if (predicted[0].numpy() == label[index]):
            accurate = accurate+1
        # else:

        print("all_data[filename]", all_data["filename"][index], "all_data[Type]", all_data["Type"][index])
        file_txt.write("all_data[filename]：" + str(all_data["filename"][index]) + "   ：" + str(all_data["Type"][index]) + '\n')
        print("top1: ", percentage_top1.numpy(), predicted.numpy(), "top2: ", percentage_top2, top2_Index,
              len(long1))        
        file_txt.write("top1:" + str(format(percentage_top1.item(), '.6f'))+ " , " + str(format(predicted.item(), '.6f'))+
                       "   top2" + str(format(percentage_top2, '.6f'))+ " , " +str(format(top2_Index, '.6f'))+" , " +
                    str(len(long1)) + '\n')
        
        #20241024解释
        # saliency_map = crcam(output_org,data_numpy)
        saliency_map = GradCam(output_org,data_numpy)
        # saliency_map = BICAM_Appendix(output_org,data_numpy)
        # saliency_map = Ablation(output_org,data_numpy)
        # saliency_map = ScoreCam(output_org,data_numpy)
        # print(data_numpy.dtype)
        # saliency_map = GradCampp(output_org,data_numpy)


        # saliency_map = ShapleyCam(predicted,data_numpy,output_org)
        total = total+1
        ave_drop, ave_increase, inse_auc, del_auc = evaluate(data_numpy, saliency_map, label,file_txt)
        ave_drop_all += ave_drop
        ave_increase_all += ave_increase
        inse_auc_all += inse_auc
        del_auc_all += del_auc
        summary(ave_drop, ave_increase, inse_auc, del_auc,count,file_txt)

        print("Data count:", count)        
        file_txt.write("Data count:" + str(format(count)) + '\n')
        gc.collect()
        count += 1
        if count >=80:
            break
    ave_drop_out = ave_drop_all / count
    ave_increase_out = ave_increase_all / count
    inse_auc_out = inse_auc_all / count
    del_auc_out = del_auc_all / count

    print("================================END==============================")
    print("总的平均下降指标：", ave_drop_out)
    print("总的平均上升指标：", ave_increase_out)
    print("总的删除曲线指标：", del_auc_out)
    print("总的插入曲线指标：", inse_auc_out)
    print("================================END==============================")
    file_txt.write("================================END==============================" + '\n')
    file_txt.write("总的平均下降指标：" + str(ave_drop_out) + '\n')
    file_txt.write("总的平均上升指标：" + str(ave_increase_out) + '\n')
    file_txt.write("总的删除曲线指标：" + str(del_auc_out) + '\n')
    file_txt.write("总的插入曲线指标：" + str(inse_auc_out) + '\n')
    file_txt.write("================================END==============================" + '\n')
    print(total)
    print(accurate)

# @torchsnooper.snoop()

def load_test_plot(all_data,label,index_,time_now):
# def load_test(all_data,label):
    accurate = 0
    # 记录结果
    # time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    # file_txt = open(f'result/crcam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/gradcam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/bicam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/ablation-{time_now}.txt', mode='a', encoding="utf-8")
    file_txt = open(f'result/scorecam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/gradcampp-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/gtcam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/wblrp-{time_now}.txt', mode='a', encoding="utf-8")
    for index in range(len(all_data["Longitude1"])):
        # if(all_data["Type"][index]!= index_):
        #     continue
        # 以下注释代码，通过index可以单测某个样本情况
        if (index!= index_):
            continue
        pretrained_model.eval()
        file_txt.write("=================================================================" + '\n')
        file_txt.write(str(datetime.datetime.now()) + '\n')
        long1 = all_data["Longitude1"][index].copy()
        long2 = all_data["Longitude2"][index].copy()
        la1 = all_data["Latitude1"][index].copy()
        la2 = all_data["Latitude2"][index].copy()
        a1 = all_data["Altitude1"][index].copy()
        a2 = all_data["Altitude2"][index].copy()

        tar0 = all_data["ReadltarRng0"][index].copy()
        tar1 = all_data["ReadltarRng1"][index].copy()
        tar2 = all_data["ReadltarRng2"][index].copy()
        tar3 = all_data["ReadltarRng3"][index].copy()

        data_val = np.zeros((1, 5, len(long1), 2, 1))
        for nums in range(min(len(long1),len(long2),len(tar1))):
            data_val[0, :, nums, 0, 0] = [long1[nums], la1[nums], a1[nums], tar0[nums], tar1[nums]]
            data_val[0, :, nums, 1, 0] = [long2[nums], la2[nums], a2[nums], tar2[nums], tar3[nums]]
            # data_numpy = np.array(data_val)
        data_numpy = data_val.astype(float)
        data_numpy = torch.Tensor(data_numpy)
        # label = label.long()
        output_org = pretrained_model(data_numpy)
        output = torch.softmax(output_org, dim=1)
        percentage_top1, predicted = torch.max(output.data, dim=1)

        percentage_top2 = np.sort(output.data.cpu().numpy())
        percentage_top2 = percentage_top2[0][-2]
        top2_Index = np.argsort(output.data.cpu().numpy())
        top2_Index = top2_Index[0][-2]
        if (predicted[0].numpy() == label[index]):
            accurate = accurate+1
        # else:
        file_txt.write("======index_：" + str(index_) + '\n')   
        print("all_data[filename]", all_data["filename"][index], "all_data[Type]", all_data["Type"][index])
        file_txt.write("all_data[filename]：" + str(all_data["filename"][index]) + "   ：" + str(all_data["Type"][index]) + '\n')
        print("top1: ", percentage_top1.numpy(), predicted.numpy(), "top2: ", percentage_top2, top2_Index,
              len(long1))        
        file_txt.write("top1:" + str(format(percentage_top1.item(), '.6f'))+ " , " + str(format(predicted.item(), '.6f'))+
                       "   top2" + str(format(percentage_top2, '.6f'))+ " , " +str(format(top2_Index, '.6f'))+" , " +
                    str(len(long1)) + '\n')
        mm = all_data["Type"][index]

        # #20241024解释
        # saliency_map = crcam(output_org,data_numpy)
        # saliency_map = GradCam(output_org,data_numpy)
        # saliency_map = BICAM_Appendix(output_org,data_numpy)
        # saliency_map = Ablation(output_org,data_numpy)
        # saliency_map = ScoreCam(output_org,data_numpy)
        # saliency_map = GradCampp(output_org,data_numpy)
        saliency_map = ShapleyCam(predicted,data_numpy,output_org)
        # skeleton_model = SkeletonLRP()
        # saliency_map = skeleton_model.lrp_func(data_numpy, str(2))
        ave_drop, ave_increase, inse_auc, del_auc = evaluate(data_numpy, saliency_map, label,file_txt)     
        print("平均下降指标：", ave_drop)
        print("平均上升指标：", ave_increase)
        print("删除曲线指标：", inse_auc)
        print("插入曲线指标：", del_auc)
        return saliency_map, predicted[0].numpy(),mm
    
def ShapleyCam(predicted,data_numpy,output_org):
        """
        In order to get the saliency_map of
        :param label:
        :param data: the original human skeleton data, shape is (N, C, T, V, M)
        :return:
        """
        p1, p2 = cut_skeleton(data_numpy)
        source_data = permutation_combination(data_numpy, predicted, p1, p2)
        shapley_value = compute_shapley_values(unique_keys, source_data)


        saliency_map = saliency_map1(shapley_value, data_numpy, output_org)
        return saliency_map

def cut_skeleton(data):
        """
        In order to make the skeleton data to five
        :param data: the skeleton data shape is (N, C, T, V, M)
        :return: five body:left_hand, right_hand, left_leg, right_leg, trunk
        """
        N, C, T, V, M = data.size()

        # 定义各个肢体的矩阵
        p1 = np.zeros((2, 5))
        p1[[0], :] = 1
        p1 = np.tile(p1, (T*M, 1))
        p1 = torch.from_numpy(p1)
        p1 = data_reshape(data, p1)

        p2 = np.zeros((2, 5))
        p2[[1], :] = 1
        p2 = np.tile(p2, (T*M, 1))
        p2 = torch.from_numpy(p2)
        p2 = data_reshape(data, p2)

        return p1, p2

def data_reshape( data, mask):
    N, C, T, V, M = data.size()
    reshaped_data = torch.reshape(mask, [N, T, V, M, C])
    reshaped_data = reshaped_data.permute(0, 4, 1, 2, 3).contiguous()

    return reshaped_data

def permutation_combination(data, label, p1, p2):
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
        pretrained_model.eval().double()
        pretrained_model.zero_grad()
        # <editor-fold desc="single. 单个一组 2 个">
        p1 = (p1 * data).float().double()
        with torch.no_grad():
            p1_out = pretrained_model(p1)
            probability_p1 = (torch.nn.functional.softmax(p1_out[0], dim=-1))[label].item()
        source_data["p1"] = probability_p1

        p2 = (p2 * data).float().double()
        with torch.no_grad():
            p2_out = pretrained_model(p2)
            probability_p2 = (torch.nn.functional.softmax(p2_out[0], dim=-1))[label].item()
        source_data["p2"] = probability_p2
        # <editor-fold desc="two. 两个一组 1个">
        data = data
        one_two = p1 + p2
        one_two = (data * one_two).float().double()
        with torch.no_grad():
            one_two_out = pretrained_model(one_two)
            probability_one_two = (torch.nn.functional.softmax(one_two_out[0], dim=-1))[label].item()
        source_data["p1,p2"] = probability_one_two
        return source_data

def compute_shapley_values(unique_keys, source_data):
        """
        In order to Compute shapley values of each permutation
        :param unique_keys: the permutation of human skeleton
        :param source_data: the probability of each permutation
        :return: the shapley values of each permutation human skeleton
        """
        # 所有的联盟
        all_coalitions = [list(j) for i in range(len(unique_keys)) for j in itertools.combinations(unique_keys, r=i + 1)]

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
           
            print('k的联盟:', k_coalition)
            print('剔除k的联盟:', no_k_coalition)

            # 遍历包含k的联盟，并计算夏普利值
            shapley_value = 0
            for i in range(len(k_coalition)):
                s = len(k_coalition[i][0])
                # 成员k的边际贡献
                k_payoff = k_coalition[i][1] - no_k_coalition[i][1]
                # 联盟的权重系数
                k_weight = math.factorial(s-1) * math.factorial(n-s) / math.factorial(n)
                shapley_value += k_payoff * k_weight

            k_shapley_value.append((k, shapley_value))
            shapley_value_list.append(k_shapley_value)

            # 联盟的权重
        print('各个部位的夏普利值：', shapley_value_list)
        return shapley_value_list

def crcam(output_org,data_numpy):
    model_output = output_org
    data = data_numpy
    # pretrained_model.train()
    pretrained_model.eval()
    activations,gradients = cfg_hook(pretrained_model)
    # if action_class is None:
    output_org = pretrained_model(data_numpy)
    action_class = model_output.argmax(dim=1, keepdim=True)
    target_one_hot = torch.zeros_like(model_output)
    target_one_hot = target_one_hot.scatter_(1, action_class, 1)

    activations = activations[-1].detach()
    # activations = []
    bs, c, t, v = activations.size()
    activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
    bs, c, t, v, m = activations.size()
    
    slope1_target = torch.zeros(bs, c)
    slope1_contrast = torch.zeros(bs, c)

    with torch.no_grad():
        for i in range(c):
            # 获取第i层特征图,并进行上采样操作
            saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
            saliency_map = F.interpolate(saliency_map,
                                            size=(data.size(2), v, m), mode='trilinear', align_corners=True)
            # 归一化
            norm_saliency_map = data_norm(saliency_map)
            ret = norm_saliency_map.mean(dim=(-3, -2, -1), keepdim=False)
            drop_data = torch.where(norm_saliency_map.repeat(1, 5, 1, 1, 1) < (ret), 
                                    torch.zeros_like(data), data)
            output = pretrained_model(drop_data)
            output=output.detach()
            # self.reset_hook_value()
            # gradients, activations = [], []
            gradients = []


            # output = F.softmax(output, dim=1)
            predict_class = output.argmax(dim=1, keepdim=True)
            pred_one_hot = torch.zeros_like(output).to(output.device)
            pred_one_hot = pred_one_hot.scatter_(1, predict_class, 1)

            pred_score = torch.sum(pred_one_hot * output, dim=-1, keepdim=False)
            target_score = torch.sum(target_one_hot * output, dim=1)

            slope1_target[:, i:i+1] = target_score
            slope1_contrast[:, i:i+1] = torch.exp(target_score - pred_score + 1)

        sorted_slope, indices_slope = torch.sort(slope1_target, dim=-1, descending=True)
        sorted_contrast_slope = torch.zeros_like(slope1_contrast)
        sorted_activations = torch.zeros_like(activations)

        for i in range(sorted_activations.size(0)):
            sorted_activations[i, :, :, :, :] = activations[i, indices_slope[i, :], :, :, :]
            sorted_contrast_slope[i, :] = slope1_contrast[i, indices_slope[i, :]]
        accum_activation = sorted_activations[:, 0:1, :, :, :]
        accm_slope = torch.zeros_like(sorted_contrast_slope)

        for i in range(1, c):
            accum_activation = accum_activation + sorted_activations[:, i:i+1, :, :, :]
            norm_accum_activation = F.interpolate(accum_activation, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
            # norm
            norm_accum_activation = data_norm(norm_accum_activation)
            ret = norm_accum_activation.mean(dim=(-3, -2, -1), keepdim=False)
            drop_data = torch.where(norm_accum_activation.repeat(1, 5, 1, 1, 1) < (ret), 
                                    torch.zeros_like(data), data)
            sub_logit= pretrained_model(drop_data)
            sub_logit=sub_logit.detach()
            # gradients, activations = [], []
            gradients = []
            # self.reset_hook_value()

            predict_class = sub_logit.argmax(dim=1, keepdim=True)
            pred_one_hot = torch.zeros_like(sub_logit).to(sub_logit.device)
            pred_one_hot = pred_one_hot.scatter_(1, predict_class, 1)
            target_score = torch.sum(target_one_hot * sub_logit, dim=-1, keepdim=False)
            pred_score = torch.sum(pred_one_hot * sub_logit, dim=-1, keepdim=False)
            accm_slope[:, i:i+1] = sorted_contrast_slope[:, i:i+1] / torch.exp(target_score - pred_score + 1)

    saliency_map = (accm_slope.view(bs, c, 1, 1, 1) * sorted_activations).sum(1, keepdim=True)
    score_saliency_map = F.relu(saliency_map)
    score_saliency_map = data_norm(score_saliency_map)
    # score_saliency_map 为所求
    score_saliency_map = F.interpolate(score_saliency_map,
                                        size=(data.size(2), v, m), mode='trilinear', align_corners=True)
    return score_saliency_map
def GradCam(output_org,data):
    '''GradCAM 方法'''
    # if self.arg.data_level == "test_set":
    #     assert self.arg.test_batch_size == 1, "Test batch size for GradCAM method must be one."
    # 预处理，获取激活和梯度
    # inference
    pretrained_model.eval()
    # pretrained_model.train()
    activations,gradients = cfg_hook(pretrained_model)
    # pretrained_model.eval()
    pretrained_model.zero_grad()

    # action recognition  
    model_output= pretrained_model(data)

    # if action_class is None:
    action_class = model_output.argmax(dim=1, keepdim=True)

    one_hot = torch.zeros_like(model_output)
    one_hot = one_hot.scatter_(1, action_class, 1)

    loss = torch.sum(one_hot * model_output)
    loss.backward(retain_graph=True)

    # 获取梯度
    gradients = gradients[0].detach()
    activations = activations[-1].detach()
    # self.reset_hook_value()

    bs, c, t, v = gradients.shape # 实际处理中bs和m已经合并
    gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
    activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
    bs, c, t, v, m = gradients.shape

    # 核心处理方法
    alpha = gradients.view(bs, c, -1).mean(2)
    weights = alpha.view(bs, c, 1, 1, 1)

    saliency_map = (weights * activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map).detach()

    saliency_map = reshape_saliency_map(saliency_map, data)

    return saliency_map
# @torchsnooper.snoop()
def saliency_map1(shapley_value, data, output_org):
        """
        In order to make the shapley value visualization and computer the evaluate
        :param shapley_value: each human skeleton shapley value
        :param data: the human skeleton data to get the each body of human
        :return: the evaluate and visualization of shapley value
        """
        p1, p2 = cut_skeleton(data)
        shapley_value = np.array(shapley_value).squeeze(1)

        saliency_map = p1 * float(shapley_value[0][1]) + p2 * float(shapley_value[1][1]) 
        base_saliency_map= GradCam(output_org, data)
        base_saliency_map = base_saliency_map.cpu()
        saliency_map = torch.sum(saliency_map, dim=1, keepdim=True).float()
        saliency_map = saliency_map * saliency_map
        saliency_map = saliency_map * base_saliency_map
        return saliency_map

def BICAM_Appendix(output_org,data):  # union OoD
        # 这里的每一个部分都是使用了softmax的，否则top1和top5很高但是increase一直为0，混淆矩阵尚未测试
        # pretrained_model.train()
        
        pretrained_model.eval()
        pretrained_model.zero_grad()
        activations,gradients = cfg_hook(pretrained_model)
        alpha=1
        k=100
        fix_ret=False
        # action recognition
        model_output  = pretrained_model(data)
        # TODO：精简化程序margin部分，支持batchsize大于一，支持多卡

        # if action_class is None:
        action_class = model_output.argmax(dim=1, keepdim=True)

        model_output = F.softmax(model_output, dim=-1)
        model_output = - torch.log(model_output)

        one_hot = torch.zeros_like(model_output)
        one_hot = one_hot.scatter_(1, action_class, 1)
        model_score = torch.sum(one_hot * model_output, dim=1).view(1, 1).contiguous().detach()
        
        activations = activations[-1].detach()
        # self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).to(self.dev).detach()
        feed_bs = 32

        # slope = torch.zeros(bs, c).double().to(self.dev)
        slope = torch.zeros(bs, c)
        with torch.no_grad():
            for i in range(0, c, feed_bs):
                
                # 获取第i层特征图,并进行上采样操作
                saliency_map = activations[:, i:i+feed_bs, :, :, :].view(feed_bs, 1, t, v, m).detach()
                saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                # 归一化
                saliency_map = data_norm(saliency_map)
                
                # ret = saliency_map.mean(dim=(-3, -2, -1), keepdim=False)
                # centre_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) <= ret, torch.zeros_like(data), data)
                # surround_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) > ret, torch.zeros_like(data), data)
                
                if fix_ret:
                    ret = 0.5
                else:
                    ret = saliency_map.mean(dim=(-3, -2, -1), keepdim=True)
                #org
                # saliency_map = saliency_map.repeat(1, 3, 1, 1, 1) - ret
                #双机编队  
                saliency_map = saliency_map.repeat(1, 5, 1, 1, 1) - ret
                #screw
                # saliency_map = saliency_map.repeat(1, 6, 1, 1, 1) - ret
                
                centre_data = torch.sigmoid(k * saliency_map) * data
                surround_data = torch.sigmoid(-k * saliency_map) * data

                del saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                pretrained_model.zero_grad()
                cen_logit  = pretrained_model(centre_data)
                cen_logit= cen_logit.detach()
                # self.reset_hook_value()
                cen_logit = F.softmax(cen_logit, dim=1)
                cen_logit = - torch.log(cen_logit)
                cen_logit = torch.sum(one_hot * cen_logit, dim=1).view(1, -1)

                pretrained_model.zero_grad()
                sur_logit  = pretrained_model(surround_data)
                cen_logit= cen_logit.detach()
                # self.reset_hook_value()
                sur_logit = F.softmax(sur_logit, dim=1)
                sur_logit = - torch.log(sur_logit)
                sur_logit = torch.sum(one_hot * sur_logit, dim=1).view(1, -1)
                
                slope[:, i:i+feed_bs] = (alpha * (model_score-cen_logit) - (1-alpha) * (model_score-sur_logit)).detach()

        slope = F.softmax(slope, dim=-1)
        
        # 归一化
        saliency_map = (slope.view(bs, c, 1, 1, 1) * activations).sum(1, keepdim=True)
        # 归一化
        score_saliency_map = data_norm(F.relu(saliency_map))
        # score_saliency_map = F.interpolate(score_saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        score_saliency_map = reshape_saliency_map(score_saliency_map, data)
        return score_saliency_map.detach()
def Ablation(output_org,data):
    # TODO：Ablation方法可以采用非简化方式，需要参考官方程序
    pretrained_model.eval()
    activations,gradients = cfg_hook(pretrained_model)
    
    pretrained_model.zero_grad()

    # action recognition
    model_output = pretrained_model(data)# tensor<(1, 120), float32, cuda:5, grad>，numclass:120
    num_classes =  model_output.shape[1]
    # if not action_class:
    action_class = model_output.argmax(dim=1, keepdim=True)#tensor<(1, 1), int64, cuda:5>

    one_hot = torch.zeros(model_output.shape, dtype=torch.float)#one_hot = tensor<(1, 120), float32, cuda:5>
    one_hot = one_hot.scatter_(1, action_class, 1)#张量上指定位置设置为1
    model_score = torch.sum(one_hot * model_output, dim=1).view(-1, 1, 1, 1, 1).contiguous()#tensor<(1, 1, 1, 1, 1), float32, cuda:5, grad>
    # TODO: 没有经过softmax的socre是否准确，需要对照原论文和源程序以及论文原始模型进行思考和调整

    activations = activations[-1].detach()#tensor<(2, 256, 11, 25), float32, cuda:5>
    # self.reset_hook_value()
    bs, c, t, v = activations.shape#bs = 2, c = 256,t = 11,v = 25
    activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
    bs, c, t, v, m = activations.shape#bs = 1, c = 256,t = 11,v = 25,m=2

    # slope = torch.zeros(bs, c, 1, 1, 1).double().to(self.dev)
    slope = torch.zeros(bs, c, 1, 1, 1)#tensor<(1, 256, 1, 1, 1), float32, cuda:5>

    with torch.no_grad():
        for i in range(c):
            pretrained_model.eval()
            pretrained_model.zero_grad()
            ablation_activations = activations.clone().detach()#tensor<(1, 256, 11, 25, 2), float32, cuda:5>
            ablation_activations = ablation_activations.permute(0, 4, 1, 2, 3).contiguous()
            ablation_activations = ablation_activations.view(bs * m, c, t, v).contiguous()
            ablation_activations[:, i:i+1, :, :] = torch.zeros(bs * m, 1, t, v)
            ablation_activations = F.avg_pool2d(ablation_activations, ablation_activations.size()[2:])
            ablation_activations = ablation_activations.view(bs, m, -1, 1, 1).mean(dim=1)#tensor<(2, 256, 1, 1), float32, cuda:5>

            # prediction
            ablation_activations = pretrained_model.fcn(ablation_activations)#forstgcn
            ablation_activations = ablation_activations.squeeze(2).squeeze(2)#
            # ablation_activations = ablation_activations.squeeze().unsqueeze(1)#for hdgcn (256,1)
            # ablation_activations = ablation_activations.expand(256, num_classes)#for hdgcn
            # ablation_activations = pretrained_model.fc(ablation_activations)#for hdgcn
            ablation_output = ablation_activations.view(ablation_activations.size(0), -1)
            ablation_score = torch.sum(one_hot * ablation_output, dim=1).view(-1, 1, 1, 1, 1).contiguous()
            # self.reset_hook_value()

            slope[:, i:i+1, :, :, :] = (model_score - ablation_score) / model_score

    saliency_map = (slope * activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)

    saliency_map = reshape_saliency_map(saliency_map, data)

    return saliency_map.detach()
def ScoreCam(output_org,data):
        pretrained_model.eval()
        activations,gradients = cfg_hook(pretrained_model)
        pretrained_model.zero_grad()
        # action recognition
        model_output = pretrained_model(data)
        # if action_class is None:
        action_class = model_output.argmax(dim=1, keepdim=True)
        one_hot = torch.zeros_like(model_output)
        one_hot = one_hot.scatter_(1, action_class, 1)

        activations = activations[-1].detach()
        # self.reset_hook_value()
        bs, c, t, v = activations.shape
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = activations.shape
        # score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m).double().to(self.dev)
        score_saliency_map = torch.zeros(bs, 1, data.size(2), v, m)

        with torch.no_grad():
            for i in range(c):
                # 获取第i层特征图,并进行上采样操作
                saliency_map = torch.unsqueeze(activations[:, i, :, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map,
                                             size=(data.size(2), v, m), mode='trilinear', align_corners=True)
                # 归一化
                norm_saliency_map = data_norm(saliency_map)
                # 利用第i层特征图作为mask覆盖原图,重新送入网络获取对应类别得分
                output= pretrained_model(data * norm_saliency_map)
                output=output.detach()
                # self.reset_hook_value()
                output = F.softmax(output, dim=1)
                score = torch.sum(one_hot * output, dim=1).view(-1, 1, 1, 1, 1).contiguous().detach()
                # 利用该得分作为权重对该层的特征图进行加权线性融合, baseline默认为全0的图,所以这里直接
                # 用该得分作为特征权重               
                score_saliency_map += (score * saliency_map).detach()
        score_saliency_map = data_norm(F.relu(score_saliency_map))
        # score_saliency_map 为所求
        return score_saliency_map.detach()
def reshape_saliency_map( saliency_map, data):
    bs, c, t, v, m = saliency_map.shape
    saliency_map = F.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
    saliency_map = data_norm(saliency_map)
    return saliency_map
# @torchsnooper.snoop()
def GradCampp(output_org,data):
        ''' 预处理，获取激活和梯度'''
        # inference
        # pretrained_model.double()
        pretrained_model.eval()
        activations,gradients = cfg_hook(pretrained_model)
        pretrained_model.zero_grad()
        # print(data.dtype)
        # action recognition

        model_output = pretrained_model(data)
        # if action_class is None:
        action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output)
        one_hot = one_hot.scatter_(1, action_class, 1)

        loss = torch.sum(one_hot * model_output)
        loss.backward(retain_graph=True)

        # 获取梯度
        gradients = gradients[0].detach()
        activations = activations[-1].detach()
        # self.reset_hook_value()

        bs, c, t, v = gradients.shape # 实际处理中bs和m已经合并
        gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = gradients.shape

        '''核心处理方法'''
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * alpha_num + \
            activations.mul(gradients.pow(3)).sum(dim=(2, 3, 4), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom+1e-6)

        relu_grad = F.relu((model_output * one_hot).sum(dim=1, keepdim=False).exp() * gradients)
        weights = (alpha * relu_grad).view(bs, c, -1).sum(-1).view(bs, c, 1, 1, 1)
        
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map).detach()
        saliency_map = reshape_saliency_map(saliency_map, data)

        return saliency_map

def evaluate(data, saliency_map, label, file_txt):
    pretrained_model = Model(**(model_args))
    drop_stride = 2
    '''评价指标'''
    if saliency_map.dtype == torch.double:
        pretrained_model = pretrained_model.double()
        data = data.double()
    pretrained_model.eval()
    pretrained_model.zero_grad()
    bs, c, t, v, m = saliency_map.shape
    data = data

    mask = gen_mask(data)
    mask = mask[0]  
    cache = saliency_map[:, :, :mask[0], :, :mask[1]].cpu()
    ## ave drop ave increase
    threshold = np.percentile(cache, 50)
    drop_data = torch.where(saliency_map.repeat(1, 5, 1, 1, 1) > threshold, 
                            data, torch.zeros_like(data))
    
    with torch.no_grad():
        pretrained_model.zero_grad()
        model_output = pretrained_model(data)
        model_output = F.softmax(model_output, dim=-1)
        # self.reset_hook_value()
        gradients, activations = [], []

        pretrained_model.zero_grad()
        drop_output = pretrained_model(drop_data)
        drop_output = F.softmax(drop_output, dim=-1)
        # self.reset_hook_value()
        gradients, activations = [], []
    one_hot = torch.zeros(model_output.shape, dtype=torch.float32)
    model_class = model_output.argmax(dim=1, keepdim=True)
    one_hot = one_hot.scatter_(1, model_class, 1)
    score = torch.sum(one_hot * model_output, dim=1)
    drop_score = torch.sum(one_hot * drop_output, dim=1)
    average_drop = (F.relu(score - drop_score) / score).sum().detach().cpu().numpy()
    increase = (score < drop_score).sum().detach().cpu().numpy()
    ## 2. insertion deletion
    drop_num = 100 // drop_stride
    threshold = [np.percentile(cache, i*drop_stride) for i in range(drop_num, 0, -1)]
    saliency_map = saliency_map.repeat(1, 5, 1, 1, 1)
    # print(threshold)

    # insersion_list = torch.zeros((1, drop_num)).double().to(self.dev)
    insersion_list = []
    
    for drop_radio in range(drop_num): 
        drop_data = torch.where(saliency_map > threshold[drop_radio], data,
                                torch.zeros_like(data)).detach()
        
        pretrained_model.zero_grad()
        with torch.no_grad():
            drop_logit = pretrained_model(drop_data)
            gradients, activations = [], []
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
        
        pretrained_model.zero_grad()
        with torch.no_grad():
            drop_logit = pretrained_model(drop_data)
            # self.reset_hook_value()
            gradients, activations = [], []
            
            drop_logit = F.softmax(drop_logit, dim=-1)
            drop_score = torch.sum(one_hot * drop_logit, dim=1)
        deletion_list.append(drop_score[0].detach().cpu().numpy())
    
    deletion_list = np.array(deletion_list)
    deletion_auc = deletion_list.sum() * drop_stride / 100
    # deletion_auc = deletion_auc.cpu().numpy()
    
    print("Insertion Curve:\n", insersion_list)
    file_txt.write("Insertion Curve:\n" + str(insersion_list) + '\n')
    print("\nInsertion AUC:", insersion_auc)
    file_txt.write("\nInsertion AUC:" + str(format(insersion_auc, '.6f')) + '\n')
    print("\nDeletion Curve:\n", deletion_list)
    file_txt.write("\nDeletion Curve:\n" + str(deletion_list) + '\n')
    print("\nDeletion AUC:", deletion_auc)
    file_txt.write("\nDeletion AUC:" + str(format(deletion_auc, '.6f')) + '\n')

    return average_drop, increase, insersion_auc, deletion_auc
def summary( ave_drop, incr, inse_auc, del_auc,count,file_txt):
    if count == 0:
        ave_drop = ave_drop
        incr = incr
        inse_auc = inse_auc
        del_auc = del_auc

    else:
        ave_drop += ave_drop
        incr += incr
        inse_auc += inse_auc
        del_auc += del_auc

    count += 1

    if count % 50 == 0:
        # print("Cross Entropy:\n", self.loss_value)
        # for i in range(len(self.topk)):
        #     print("Top", self.topk[i], ":", self.top_k[:, i],
        #           "Batch size:", self.batch_size,
        #           "Count:", self.count,
        #           "Data count:", self.data_set_len)
        print("Average Drop:", ave_drop, 
            ", Increase:", incr, 
            ", Insersion AUC:", inse_auc, 
            ", Deletion AUC:", del_auc,
            )
        file_txt.write("Average Drop:" + str(format(ave_drop, '.6f')) +"; " + "Increase:" + str(format(incr, '.6f'))+'\n')
        file_txt.write("Insersion AUC:" + str(format(inse_auc, '.6f')) +"; " + "Deletion AUC:" + str(format(del_auc, '.6f'))+'\n')
def gen_mask( data): 
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
# @torchsnooper.snoop()
def cfg_hook(pretrained_model):
    gradients, activations = [], []

    def backward_hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook_fn(module, feat_in, feat_out):
        activations.append(feat_out.detach())

    def input_backward_hook_fn(module, feat_in, feat_out):
        input_grad = feat_out

    modules = list(pretrained_model.modules())
    for module in modules:
        if isinstance(module, torch.nn.ReLU):
            module.register_forward_hook(forward_hook_fn)
            module.register_backward_hook(backward_hook_fn)
        if isinstance(module, torch.nn.Conv2d):
            module.register_backward_hook(input_backward_hook_fn)

    return activations, gradients
def data_norm(data):
    bs, c, t, v, m = data.shape
    data_min = data.view(bs, -1).min(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
    data_max = data.view(bs, -1).max(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
    # denominator = torch.where(
    #     (data_max-data_min) != 0., data_max-data_min, torch.tensor(1.).double().to(self.dev))
    denominator = torch.where(
        (data_max-data_min) != 0., data_max-data_min, torch.ones_like(data_max))

    return (data - data_min) / denominator
def load_test_single(all_data,label):
    accurate = 0
    total = 0

    indexall = [16,17,19,20,21,23,24,25,29,30,31,34,35,41,42,43,44,45,46]
    # indexall = [ 44]    #zongdui 0,3,7,12,16,23,26,41,44
    indexall = [25 ,27] #hengdui  25 27
    # indexall = [9,13,18,33,39,47] #9,13,18,33,39,47
    # indexall = [17,21,34,40,46]  #双机舒开 9,13,18,33,39,47
    # indexall = [11, 19, 31, 32, 49]  # 双机包夹 9,13,18,33,39,47
    for index in indexall:
        avglat = 0
        avglon = 0
        # index = index + 235
        long1 = all_data["Longitude1"][index].copy()
        long2 = all_data["Longitude2"][index].copy()
        la1 = all_data["Latitude1"][index].copy()
        la2 = all_data["Latitude2"][index].copy()
        a1 = all_data["Altitude1"][index].copy()
        a2 = all_data["Altitude2"][index].copy()

        tar0 = all_data["ReadltarRng0"][index].copy()
        tar1 = all_data["ReadltarRng1"][index].copy()
        tar2 = all_data["ReadltarRng2"][index].copy()
        tar3 = all_data["ReadltarRng3"][index].copy()

        start = 0
        for end in range(1,len(all_data["Longitude1"][index])):
            data_val = np.zeros((1, 5, end, 2, 1))
            for nums in range(end):

                data_val[0, :, nums, 0, 0] = [long1[start:end][nums], la1[start:end][nums], a1[start:end][nums], tar0[start:end][nums], tar1[start:end][nums]]
                data_val[0, :, nums, 1, 0] = [long2[start:end][nums], la2[start:end][nums], a2[start:end][nums], tar2[start:end][nums], tar3[start:end][nums]]
                # data_numpy = np.array(data_val)
            if end < 9:
                continue
            curdistance = getDistance(la1[start:end][-1],long1[start:end][-1],
                                           a1[start:end][-1], la2[start:end][-1],
                                           long2[start:end][-1], a1[start:end][-1])

            # pitch1 = getElevation(long1[start:end][-2], la1[start:end][-2], a1[start:end][-2], long1[start:end][-1],la1[start:end][-1], a1[start:end][-1])
            # pitch2 = getElevation(long2[start:end][-2],la2[start:end][-2], a2[start:end][-2],long2[start:end][-1],la2[start:end][-1], a2[start:end][-1])
            # pitch3 = getElevation(long1[start:end][-3 - 5], la1[start:end][-3- 5], a1[start:end][-3- 5], long1[start:end][-2- 5],la1[start:end][-2- 5], a1[start:end][-2- 5])
            # pitch4 = getElevation(long2[start:end][-3 - 5], la2[start:end][-3- 5], a2[start:end][-3- 5], long2[start:end][-2- 5],la2[start:end][-2- 5], a2[start:end][-2- 5])

            # azi1 = lonlat2Azimuth(long1[start:end][-2], la1[start:end][-2],long1[start:end][-1], la1[start:end][-1])
            # azi2 = lonlat2Azimuth(long2[start:end][-2], la2[start:end][-2],long2[start:end][-1], la2[start:end][-1])
            # azi3 = lonlat2Azimuth(long1[start:end][-3 - 5], la1[start:end][-3 - 5], long1[start:end][-2 - 5], la1[start:end][-2 - 5])
            # azi4 = lonlat2Azimuth(long2[start:end][-3 - 5], la2[start:end][-3 - 5], long2[start:end][-2 - 5], la2[start:end][-2 - 5])
          

            data_numpy = data_val.astype(np.float)

            data_numpy = torch.Tensor(data_numpy)
            # label = label.long()
            output = pretrained_model(data_numpy)
            output = torch.softmax(output, dim=1)
            percentage_top1, predicted = torch.max(output.data, dim=1)

            percentage_top2 = np.sort(output.data.cpu().numpy())
            percentage_top2 = percentage_top2[0][-2]
            top2_Index = np.argsort(output.data.cpu().numpy())
            top2_Index = top2_Index[0][-2]
            if (predicted[0].numpy() == label[index]):
                accurate = accurate+1
            # print("all_data[filename]", all_data["filename"][index], "all_data[Type]", all_data["Type"][index])
            print("top1: ", percentage_top1.numpy(), predicted.numpy(), "top2: ", percentage_top2, top2_Index,len(long1))
            total = total+1

        print(all_data["filename"][index], ": ", "True:", all_data["Type"][index], "predict: ", predicted[0].numpy() ,"Frames:",total, "acc:",accurate/total)
        total = 0
        accurate = 0
        # print(total)
        # print(accurate)
        # print(accurate/total)
def getDistance(lat1, lng1, h1, lat2, lng2, h2):
    # radLat1 = rad(lat1)
    # radLat2 = rad(lat2)
    a = lat1 - lat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat1) * math.cos(lat2) * math.pow(math.sin(b / 2), 2)))
    s = s * 6378.137
    s = math.sqrt(s * s + (h1 - h2) * (h1 - h2))
    return s
def rad(d):
    return d * math.pi / 180.0
def hav(theta):
    s = math.sin(theta / 2)
    return s * s

def get_distance_hav(lat1, lng1, lat2, lng2):
    """用haversine公式计算球面两点间的距离。"""
    # 经纬度转换成弧度
    # lat0 = math.radians(lat0)
    # lat1 = math.radians(lat1)
    # lng0 = math.radians(lng0)
    # lng1 = math.radians(lng1)
    #
    # dlng = math.fabs(lng0 - lng1)
    # dlat = math.fabs(lat0 - lat1)
    # h = hav(dlat) + math.cos(lat0) * math.cos(lat1) * hav(dlng)
    # distance = 2 * 6378.137 * math.asin(math.sqrt(h))
    # radLat1 = rad(lat1)
    # radLat2 = rad(lat2)

    R = 6378137
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return c*10000
def wgs84ToNED(lat, lon, h, lat0=24.8976763, lon0=160.123456, h0=0):
    # function[xEast, yNorth, zUp] = geodetic_to_enu(lat, lon, h, lat0, lon0, h0)
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)

    lamb = math.radians(lat)  # 角度换成弧度
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    # 原点坐标转换
    lamb0 = math.radians(lat0)
    phi0 = math.radians(lon0)
    s0 = math.sin(lamb0)
    N0 = a / math.sqrt(1 - e_sq * s0 * s0)

    sin_lambda0 = math.sin(lamb0)
    cos_lambda0 = math.cos(lamb0)
    sin_phi0 = math.sin(phi0)
    cos_phi0 = math.cos(phi0)

    x0 = (h0 + N0) * cos_lambda0 * cos_phi0
    y0 = (h0 + N0) * cos_lambda0 * sin_phi0
    z0 = (h0 + (1 - e_sq) * N0) * sin_lambda0

    xd = x - x0
    yd = y - y0
    zd = z - z0

    t = -cos_phi0 * xd - sin_phi0 * yd

    xEast = -sin_phi0 * xd + cos_phi0 * yd
    yNorth = t * sin_lambda0 + cos_lambda0 * zd
    zUp = cos_lambda0 * cos_phi0 * xd + cos_lambda0 * sin_phi0 * yd + sin_lambda0 * zd

    return yNorth, xEast, -zUp
    # return xEast, yNorth, zUp

def calculate_direction(positions):
    """
    计算飞行方向向量
    positions: 飞机的位置列表，每个位置是经纬度坐标的元组
    """
    start = np.array(positions[0])
    end = np.array(positions[-1])
    direction_vector = end - start
    norm = np.linalg.norm(direction_vector)
    # if norm == 0:
    #     return None
    return direction_vector / norm
def calculate_relative_angle(vector1, vector2):
    """
    计算两个向量之间的夹角
    vector1: 第一个向量
    vector2: 第二个向量
    """
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cos_theta)
    return np.degrees(angle)  # 将弧度转换为角度

# 判断队形
def determine_formation(angle):
    """
    通过角度判断队形
    angle: 飞机间的相对角度
    """
    epsilon = 30.0  # 设置一个容差值，10度以内的角度变化不改变队形
    if (180 - epsilon) <= angle <= (180 + epsilon) or angle <= epsilon or 360 - epsilon <= angle <= 360 :
        return "横队"
    elif (90 - epsilon) <= angle <= (90 + epsilon) or (270 - epsilon) <= angle <= (270 + epsilon):
        return "纵队"
    else:
        return "梯队"

# 计算两点之间的距离和方向
def calculate_distance_and_bearing(aircraft1, aircraft2):
    # 假设为简单的欧式距离计算，实际应用中可能需要更复杂的大圆距离计算
    lat1, lon1 = np.radians(aircraft1.latitude), np.radians(aircraft1.longitude)
    lat2, lon2 = np.radians(aircraft2.latitude), np.radians(aircraft2.longitude)

    # 计算方位角
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    initial_bearing = np.arctan2(x, y)

    # 方向角度转换为正北方向的度数
    bearing = (np.degrees(initial_bearing) + 360) % 360

    return bearing

def getElevation(lonA, latA, altA, lonB, latB, altB):
    dLon = lonB - lonA
    dLat = latB - latA
    dAlt = altB - altA

    distance = lonlat2dist(lonA, latA, lonB, latB)
    pitch = np.arctan(dAlt / distance)
    pitch = np.degrees(pitch) #将弧度转为度
    return pitch

def lonlat2dist(lonA, latA, lonB, latB):
    R = 6371393
    toRad = np.pi / 180

    cosC = np.cos((90 - latB) * toRad) * np.cos((90 - latA) * toRad) + np.sin((90 - latB) * toRad) * np.sin(
        (90 - latA) * toRad) * np.cos((lonB - lonA) * toRad)

    C = np.arccos(cosC)
    return C * R

# 计算偏航角
def lonlat2Azimuth(lonA, latA, lonB, latB):

    toRad = np.pi / 180

    cosC = np.cos((90 - latB) * toRad) * np.cos((90 - latA) * toRad) + np.sin((90 - latB) * toRad) * np.sin((90 - latA) * toRad) * np.cos((lonB - lonA) * toRad)
    sinC = np.sqrt(1 - np.square(cosC))
    sinA = np.sin((90 - latB) * toRad) * np.sin((lonB - lonA) * toRad) / sinC

    if latA > latB:
        return  (180 - np.arcsin(sinA) / toRad)
    return (np.arcsin(sinA) / toRad)




