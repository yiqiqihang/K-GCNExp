import math
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.format import open_memmap
from GNCData import  wgs84ToNED
from collections import Counter
from Model import load_model, load_test, load_test_single
from getCsvdata import get_csv_data


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

    ReadltarRng0 = []
    ReadltarRng1 = []
    ReadltarRng2 = []
    ReadltarRng3 = []
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
        f1 = np.reshape(f1, [5, 601])
        f2 = np.reshape(f2, [5, 601])
        f1 = f1.T
        f2 = f2.T
        lon1 = []
        lon2 = []
        lat1 = []
        lat2 = []
        h1 = []
        h2 = []
        realRng1 = []
        realRng2 = []
        realRng3 = []
        realRng4 = []
        # print(f1)
        for j in range(601):
            # if f1[j][0] == 0:
            #     continue
            lo1 = f1[j][0]
            lo2 = f2[j][0]
            la1 = f1[j][1]
            la2 = f2[j][1]
            H1 = f1[j][2]
            H2 = f2[j][2]
            # r1 = f1[j][3]
            # r2 = f1[j][4]
            # r3 = f2[j][3]
            # r4 = f2[j][3]

            # use_xyz = True
            if isxyz:
                lo1,la1,H1 = wgs84ToNED(float(la1), float(lo1), float(H1))
                lo2,la2,H2 = wgs84ToNED(float(la2), float(lo2), float(H2))
            lon1.append(lo1)
            lon2.append(lo2)
            lat1.append(la1)
            lat2.append(la2)
            h1.append(H1)
            h2.append(H2)
            realRng1.append(f1[j][3])
            realRng3.append(f2[j][3])
            realRng2.append(f1[j][4])
            realRng4.append(f2[j][4])
        Longitude1.append(lon1)
        Latitude1.append(lat1)
        Altitude1.append(h1)
        Longitude2.append(lon2)
        Latitude2.append(lat2)
        Altitude2.append(h2)

        ReadltarRng0.append(realRng1)
        ReadltarRng1.append(realRng2)
        ReadltarRng2.append(realRng3)
        ReadltarRng3.append(realRng4)

    # print(len(lon1))
    allData['filename'] = filen_name
    allData['Type'] = Type

    allData['Longitude1'] = Longitude1
    allData['Latitude1'] = Latitude1
    allData['Altitude1'] = Altitude1
    allData['Longitude2'] = Longitude2
    allData['Latitude2'] = Latitude2
    allData['Altitude2'] = Altitude2

    allData['ReadltarRng0'] = ReadltarRng0
    allData['ReadltarRng1'] = ReadltarRng1
    allData['ReadltarRng2'] = ReadltarRng2
    allData['ReadltarRng3'] = ReadltarRng3

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

def plot(data):
    La1 = []
    Long1 = []
    A1 = []
    La2 = []
    Long2 = []
    A2 = []

    for j in range(0,601,2):
        if (data[0][j][0] == 0):
            continue
        La1.append(data[0][j][0])
        Long1.append(data[1][j][0])
        A1.append(data[2][j][0])

    for j in range(601):
        print(data[0][j][0])
        if (data[0][j][0] == 0):
            continue
        La2.append(data[0][j][1])
        Long2.append(data[1][j][1])
        A2.append(data[2][j][1])

    fig = plt.figure(dpi=1024)
    # ax = fig.add_subplot()
    ax = fig.add_subplot(111, projection='3d')
    color = 'r'
    cut = 10
    figure0 = ax.scatter(La1[::cut], Long1[::cut], A1[::cut], c=color, marker='o', s=1)
    figure1 = ax.scatter(La2[::cut], Long2[::cut], A2[::cut], c=color, marker='o', s=1)
    plt.show()


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

    # 打印结果
    for value, count in counted_values.items():
        print(f"{value}: {count}")






if __name__ == '__main__':
    # data_path = "enhance_train_data_plane_Radar.npy"
    # label_path = "enhance_train_label_plane_Radar.pkl"
    # data_path = "enhance_train_data_plane_Radar_0328_300Frames.npy"
    # label_path = "enhance_train_label_plane_Radar_0328_300Frames.pkl"
    data_path = "rechange_val_data_plane_Radar.npy"
    label_path = "rechange_val_label_plane_Radar.pkl"


    isxyz = False
    data = np.load(data_path, mmap_mode='r')
    with open(label_path, 'rb') as f:
        filename, label = pickle.load(f)
    # print(len(filename))
    # 获取数据
    # all_data = get_data(data,isxyz,filename,label)

    #打印每种战术的数量
    # caculate_nums(label)

    
    plot(data[0])

    # 测试集中每个样本的识别情况
    # load_test(all_data,label)


    # 每帧情况
    # load_test_single(all_data,label)

    # 添加特征，双机的俯仰角与偏航角
    # all_data = enhanceData.enhance_azi_pitch(all_data)

    print(len(label))
    print("------------------")

