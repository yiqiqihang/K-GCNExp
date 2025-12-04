import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from geopy.distance import geodesic
from geopy.point import Point
from net.st_gcn import Model
# 加载预训练模型的特征提取部分
model_args = {'in_channels': 5, 'num_class': 10, 'dropout': 0, 'edge_importance_weighting': False,
              'graph_args': {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}} 


pre_model_path =  r"models\STGCN_model.pt"
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


def load_test(all_data,label):
    accurate = 0
    total = 0
    for index in range(len(all_data["Longitude1"])):
        # if(all_data["Type"][index]!= 3 ):
        #     continue

        # 以下注释代码，通过index可以单测某个样本情况
        # if (index!=2):
        #     continue
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
        # else:
        print("all_data[filename]", all_data["filename"][index], "all_data[Type]", all_data["Type"][index])
        print("top1: ", percentage_top1.numpy(), predicted.numpy(), "top2: ", percentage_top2, top2_Index,
              len(long1))
        total = total+1
        # if(predicted.numpy() != 7):
        # print(index)

        # # # 画图验证
        # fig = plt.figure()
        # # ax = fig.add_subplot()
        # ax = fig.add_subplot(111, projection='3d')
        # color = 'r'
        # figure0 = ax.scatter(la1, long1, a1, c=color, marker='o', s=1)
        # figure1 = ax.scatter( la2,long2, a2, c=color, marker='o', s=1)
        # saveName = "test3d"
        # # plt.savefig(saveName)
        # plt.show()
        # fig1 = plt.figure()
        # ax = fig1.add_subplot()
        # color = 'r'
        # figure0 = ax.scatter(la1, long1, c=color, marker='o', s=1)
        # figure1 = ax.scatter(la2, long2, c=color, marker='o', s=1)
        # saveName = "test2d"
        # # plt.savefig(saveName)
        # plt.show()
        # # plt.close()
    print(total)
    print(accurate)
    # print(accurate/total)


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




def load_test(all_data,label,index_,time_now):
# def load_test(all_data,label):
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
    # time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    # # file_txt = open('result-forcam-20240304/' + "ntu60-xsub" + time_now + '.txt', mode='a', encoding="utf-8")
    file_txt = open(f'result/crcam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/gradcam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/bicam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/ablation-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/scorecam-{time_now}.txt', mode='a', encoding="utf-8")
    # file_txt = open(f'result/gradcampp-{time_now}.txt', mode='a', encoding="utf-8")
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
        # file_txt.write("======index_：" + str(index_) + '\n')   
        # print("all_data[filename]", all_data["filename"][index], "all_data[Type]", all_data["Type"][index])
        # file_txt.write("all_data[filename]：" + str(all_data["filename"][index]) + "   ：" + str(all_data["Type"][index]) + '\n')
        # print("top1: ", percentage_top1.numpy(), predicted.numpy(), "top2: ", percentage_top2, top2_Index,
        #       len(long1))        
        # file_txt.write("top1:" + str(format(percentage_top1.item(), '.6f'))+ " , " + str(format(predicted.item(), '.6f'))+
        #                "   top2" + str(format(percentage_top2, '.6f'))+ " , " +str(format(top2_Index, '.6f'))+" , " +
        #             str(len(long1)) + '\n'
            file_txt.write("======index_：" + str(index_) + '\n')   
            print("all_data[filename]", all_data["filename"][index], "all_data[Type]", all_data["Type"][index])
            file_txt.write("all_data[filename]：" + str(all_data["filename"][index]) + "   ：" + str(all_data["Type"][index]) + '\n')
            print("top1: ", percentage_top1.numpy(), predicted.numpy(), "top2: ", percentage_top2, top2_Index,
                len(long1))        
            file_txt.write("top1:" + str(format(percentage_top1.item(), '.6f'))+ " , " + str(format(predicted.item(), '.6f'))+
                        "   top2" + str(format(percentage_top2, '.6f'))+ " , " +str(format(top2_Index, '.6f'))+" , " +
                        str(len(long1)) + '\n')
            #20241024解释
            saliency_map = crcam(output_org,data_numpy)
            # saliency_map = GradCam(output_org,data_numpy)
            # saliency_map = BICAM_Appendix(output_org,data_numpy)
            # saliency_map = Ablation(output_org,data_numpy)
            # saliency_map = ScoreCam(output_org,data_numpy)
            # saliency_map = GradCampp(output_org,data_numpy)

        # #20241024解释
        # saliency_map = crcam(output_org,data_numpy)
        # # saliency_map = GradCam(output_org,data_numpy)
        # # saliency_map = BICAM_Appendix(output_org,data_numpy)
        # # saliency_map = Ablation(output_org,data_numpy)
        # # saliency_map = ScoreCam(output_org,data_numpy)
        # # saliency_map = GradCampp(output_org,data_numpy)
        
        # total = total+1
  
        # # ave_drop, ave_increase, inse_auc, del_auc = evaluate(data_numpy, saliency_map, label,file_txt)
        # # ave_drop_all += ave_drop
        # # ave_increase_all += ave_increase
        # # inse_auc_all += inse_auc
        # # del_auc_all += del_auc
        # # summary(ave_drop, ave_increase, inse_auc, del_auc,count,file_txt)

        # print("Data count:", count)        
        # file_txt.write("Data count:" + str(format(count)) + '\n')

        # gc.collect()
        # count += 1
        # if count >=80:
        #     break
    # ave_drop_out = ave_drop_all / count
    # ave_increase_out = ave_increase_all / count
    # inse_auc_out = inse_auc_all / count
    # del_auc_out = del_auc_all / count

    # print("================================END==============================")
    # print("总的平均下降指标：", ave_drop_out)
    # print("总的平均上升指标：", ave_increase_out)
    # print("总的删除曲线指标：", del_auc_out)
    # print("总的插入曲线指标：", inse_auc_out)
    # print("================================END==============================")
    # file_txt.write("================================END==============================" + '\n')
    # file_txt.write("总的平均下降指标：" + str(ave_drop_out) + '\n')
    # file_txt.write("总的平均上升指标：" + str(ave_increase_out) + '\n')
    # file_txt.write("总的删除曲线指标：" + str(del_auc_out) + '\n')
    # file_txt.write("总的插入曲线指标：" + str(inse_auc_out) + '\n')
    # file_txt.write("================================END==============================" + '\n')
        # if(predicted.numpy() != 7):
        # print(index)
    # print(total)
    # print(accurate)
    # print(accurate/total)
    return saliency_map, label[index]