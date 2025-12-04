# coding=utf-8
import math
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.format import open_memmap
from GNCData import  wgs84ToNED
from collections import Counter
# from Model import load_model, load_test, load_test_single #识别
from getCsvdata import get_csv_data
from Model_crcam import load_model, load_test, load_test_single,load_test_noplot#解释
import time
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

def convert_data_to_dict(data,filename, label):
    # 初始化空字典
    allData = {}
    filen_name = []
    Type = []

    allData['filename'] = filen_name
    allData['Type'] = Type

    for i in range(len(data)):
        filen_name.append(filename[i])
        Type.append(label[i])
    
    # 从数据中提取维度
    num_samples, num_locations, num_frames, num_planes, _ = data.shape
    
    # 为每架飞机的经纬度和高度创建键
    for i in range(num_planes):
        allData[f'Longitude{i+1}'] = data[:, 0, :, i, 0]  # 经度
        allData[f'Latitude{i+1}'] = data[:, 1, :, i, 0]   # 纬度
        allData[f'Altitude{i+1}'] = data[:, 2, :, i, 0]   # 高度
    
    return allData


def get_data(data,isxyz,filename,label):
    allData = {}
    filen_name = []
    Type = []

    Longitude1 = []
    Latitude1 = []
    Altitude1 = []
    Longitude2 = []
    Latitude2 = []
    Altitude2 = []

    # 获取标签
    print(len(data))
    # print(len(filename))
    for i in range(len(data)):
        filen_name.append(filename[i])
        Type.append(label[i])

    #重新reshape数据
    data_shape = data.shape
    data = np.reshape(data, [data_shape[0], data_shape[1], data_shape[2], data_shape[3]])

    # 遍历
    for i in data:
        f1, f2 = np.split(i, 2, axis=2)
        f1 = np.reshape(f1, [3, 301])
        f2 = np.reshape(f2, [3, 301])
        f1 = f1.T
        f2 = f2.T
        lon1 = []
        lon2 = []
        lat1 = []
        lat2 = []
        h1 = []
        h2 = []

        for j in range(301):

            lo1 = f1[j][0]
            lo2 = f2[j][0]
            la1 = f1[j][1]
            la2 = f2[j][1]
            H1 = f1[j][2]
            H2 = f2[j][2]
            
            if isxyz:
                lo1,la1,H1 = wgs84ToNED(float(la1), float(lo1), float(H1))
                lo2,la2,H2 = wgs84ToNED(float(la2), float(lo2), float(H2))
            lon1.append(lo1)
            lon2.append(lo2)
            lat1.append(la1)
            lat2.append(la2)
            h1.append(H1)
            h2.append(H2)

        Longitude1.append(lon1)
        Latitude1.append(lat1)
        Altitude1.append(h1)
        Longitude2.append(lon2)
        Latitude2.append(lat2)
        Altitude2.append(h2)



    # print(len(lon1))
    allData['filename'] = filen_name
    allData['Type'] = Type

    allData['Longitude1'] = Longitude1
    allData['Latitude1'] = Latitude1
    allData['Altitude1'] = Altitude1
    allData['Longitude2'] = Longitude2
    allData['Latitude2'] = Latitude2
    allData['Altitude2'] = Altitude2

    

    return allData


def read_xyz_test(seq_info,minFrame,typeNum):
    data = np.zeros((7, minFrame, 2, 1))

    # print(seq_info['Altitude1'][typeNum])
    # print(seq_info['ReadltarRng0'][typeNum])
    for nums in range(minFrame):
        data[:, nums, 0, 0] = [seq_info['Longitude1'][typeNum][nums], seq_info['Latitude1'][typeNum][nums],
                               seq_info['Altitude1'][typeNum][nums],seq_info['ReadltarRng0'][typeNum][nums],
                               seq_info['ReadltarRng1'][typeNum][nums],
                               # seq_info['yaw1'][typeNum][nums],
                               seq_info['pitch1'][typeNum][nums], seq_info['azi1'][typeNum][nums],
                               ]  # n表示第几帧，j表示第几个节点，m表示共有几个人
        data[:, nums, 1, 0] = [seq_info['Longitude2'][typeNum][nums], seq_info['Latitude2'][typeNum][nums],
                               seq_info['Altitude2'][typeNum][nums],seq_info['ReadltarRng2'][typeNum][nums],
                               seq_info['ReadltarRng3'][typeNum][nums],
                               # seq_info['yaw2'][typeNum][nums],
                               seq_info['pitch2'][typeNum][nums], seq_info['azi2'][typeNum][nums],
                               # seq_info['Airspeed2'][typeNum][nums],
                               # seq_info['State'][typeNum][nums],
                               ]  # n表示第几帧，j表示第几个节点，m表示共有几个人
    return data





def plot(data,saliency_map):
    time_now1 = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
    La1 = []
    Long1 = []
    A1 = []

    
    A2 = []
    La2 = []
    Long2 = []
    
    A3 = []
    La3 = []
    Long3 = []
    
    A4 = []
    La4 = []
    Long4 = []

    A5 = []
    La5 = []
    Long5 = []

    fig = plt.figure(dpi=1024)

    ax = fig.add_subplot(111, projection='3d')
   
    # _, T, V, M = pose.shape#原始数据
    saliency_map = saliency_map[0][0]#mask
    z_max = 0
    #for wplrp and gtcam
    fea=saliency_map
    fea = np.clip(fea, 0, 1)
    #for all not wplrp
    # fea = ((saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-12))
    for j in range(0,300):
        if (data[0][j][0] == 0):
            continue
        La1.append(float(data[0][j][0]))
        Long1.append(float(data[1][j][0]))
        A1.append(float(data[2][j][0]))
    for j in range(0,300):
        if (data[0][j][1] == 0):
            continue
        La2.append(float(data[0][j][1]))
        Long2.append(float(data[1][j][1]))
        A2.append(float(data[2][j][1]))
        # ax.scatter3D(data[0][j][1],data[1][j][1],data[2][j][1], color='red', s=20)

    for j in range(0,300):
        if (data[0][j][2] == 0):
            continue
        La3.append(float(data[0][j][2]))
        Long3.append(float(data[1][j][2]))
        A3.append(float(data[2][j][2]))

    for j in range(0,300):
        if (data[0][j][3] == 0):
            continue
        La4.append(float(data[0][j][3]))
        Long4.append(float(data[1][j][3]))
        A4.append(float(data[2][j][3]))

    for j in range(0,300):
        if (data[0][j][4] == 0):
            continue
        La5.append(float(data[0][j][4]))
        Long5.append(float(data[1][j][4]))
        A5.append(float(data[2][j][4]))
 
    cut = 3
    color1 = [0, 0.4470, 0.7410]
    ax.text(0, 0, 0, "O (0,0,0)", color='black', fontsize=12,
    verticalalignment='bottom', horizontalalignment='left', zorder=5)

    # 调整视角，确保能看到原点标注
    ax.view_init(elev=30, azim=-45)

    color2 = [[0.4660 ,0.6740, 0.1880]]
    for j in range(0,60,cut):
        f = fea[j, :]
        ax.scatter3D(La1[j],Long1[j],A1[j], color='blue', s=40, marker='>')
        ax.scatter3D(La1[j],Long1[j],A1[j], color='red', s=40, marker='o',alpha=f[0, 0].item())

    for j in range(61,120,cut):
        f = fea[j, :]
        ax.scatter3D(La1[j],Long1[j],A1[j], color='black', s=40, marker='>')
        ax.scatter3D(La1[j],Long1[j],A1[j], color='red', s=40, marker='o',alpha=f[0, 0].item())

    for j in range(121,180,cut):
        f = fea[j, :]
        ax.scatter3D(La1[j],Long1[j],A1[j], color='purple', s=40, marker='>')
        ax.scatter3D(La1[j],Long1[j],A1[j], color='red', s=40, marker='o',alpha=f[0, 0].item())

    for j in range(181,240,cut):
        f = fea[j, :]
        ax.scatter3D(La1[j],Long1[j],A1[j], color='green', s=40, marker='>')
        ax.scatter3D(La1[j],Long1[j],A1[j], color='red', s=40, marker='o',alpha=f[0, 0].item())

    for j in range(241,len(Long1)-1,cut):
        f = fea[j, :]
        ax.scatter3D(La1[j],Long1[j],A1[j], color='dimgray', s=40, marker='>')
        ax.scatter3D(La1[j],Long1[j],A1[j], color='red', s=40, marker='o',alpha=f[0, 0].item())

    #标记初始节点
    # ax.text(La1[0], Long1[0], A1[0], 'Start', color='red')
    # ax.text(La1[-2], Long1[-2], A1[-2], 'End', color='green')

    # print("len(La1)",La1[-1])
    # print("len(Long1)",Long1[-1])
    # print("len(A1)",A1[-1])
    # print("len(La2)",La2[-1])
    # print("len(Long2)",Long2[-1])

    
    # ax.set_ylabel(' time\n----->', fontsize=20)
    ax.set_ylabel(' time\n<-----', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel('time')
    print("saveing")
    if not os.path.exists(visoutput_result_dir):
        os.makedirs(visoutput_result_dir)
        # print("save_dir",visoutput_result_dir)
    # plt.savefig('{}/Label{}+{}-{}.svg'.format(visoutput_result_dir,Typelabel,Typelabel1, time_now1))
    plt.savefig('{}/Label{}+{}-{}.png'.format(visoutput_result_dir,Typelabel,Typelabel1, time_now1))
    #Typelabel预测标签,Typelabel1是真实标签

def caculate_nums(label):
    index = []
    train_label = []
    # 获取相同战术的索引值
    for k in range(10):
        index.append([i for i, val in enumerate(label) if val == k])
    for k in range(10):
        for a in range(len(index[k])):
            train_label.append(k)

    # 使用Counter计算相同值的个数
    counted_values = Counter(train_label)

    # # 打印结果
    # for value, count in counted_values.items():
    #     print(f"{value}: {count}")




if __name__ == '__main__':
    # data_path = "enhance_train_data_plane_Radar.npy"
    # label_path = "enhance_train_label_plane_Radar.pkl"
    # data_path = "enhance_train_data_plane_Radar_0328_300Frames.npy"
    # label_path = "enhance_train_label_plane_Radar_0328_300Frames.pkl"
    data_path = "/data/home/st/GT_CAM/st-gcn/test_fiveairplanes-v0.1/train_data_plane_301Frames.npy"
    label_path = "/data/home/st/GT_CAM/st-gcn/test_fiveairplanes-v0.1/train_label_plane_301Frames.pkl"
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())

    isxyz = False
    data = np.load(data_path, mmap_mode='r')
    with open(label_path, 'rb') as f:
        filename, label = pickle.load(f)
    # print(len(filename))
    # 获取数据
    all_data = convert_data_to_dict(data,filename,label)
    # method = "scorecam"
    #打印每种战术的数量
    # caculate_nums(label)

    # # 测试集中每个样本的识别情况，批量计算精度
    # # 批量解释
    # for index in range(70):
    # load_test_noplot(all_data,label)

    # 测试集中每个样本的识别情况，批量可视化重要性
    # visoutput_result_dir =f'./result/visualization/crcam-{time_now}/'
    # visoutput_result_dir =f'./result/visualization/gradcam-{time_now}/'
    # visoutput_result_dir =f'./result/visualization/bicam-{time_now}/'
    # visoutput_result_dir =f'./result/visualization/ablationcam-{time_now}/'
    # visoutput_result_dir =f'./result/visualization/scorecam-{time_now}/'
    #visoutput_result_dir =f'./result/visualization/gradcampp-{time_now}/'
    visoutput_result_dir =f'/data/home/st/GT_CAM/st-gcn/test_fiveairplanes-v0.1/result/visualization/gtcam-{time_now}/'
    # visoutput_result_dir =f'./result/visualization/wblrp-{time_now}/'
    # index = 0
    # for index in range(90):
    # for index in range(0, 12):
    # for index in range(12, 24):
    # for index in range(24, 34):#24-33
    # for index in range(34, 45):#34-43
    # for index in range(44, 55):#44-53
    # for index in range(44, 55):#44-53
    # for index in range(55, 65):#55-63#index55ok
    # for index in range(10, 60):#10-58
    # for index in range(59, 200):#
    # 指定索引列表
    # indices = [23, 49, 190]
    indices = [23,44]
    # 遍历指定的索引
    # for index in range(90):
    for index in indices:
        saliencymap,Typelabel,Typelabel1 = load_test(all_data,label,index,time_now)
        plot(data[index],saliencymap)
        print("======index",index)


    # 每帧情况
    # load_test_single(all_data,label)

    # 添加特征，双机的俯仰角与偏航角
    # all_data = enhanceData.enhance_azi_pitch(all_data)+

    # print(len(label))
    # print("------------------")

