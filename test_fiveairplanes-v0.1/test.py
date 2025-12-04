import numpy as np

def convert_data_to_dict(data):
    # 初始化空字典
    allData = {}
    
    # 从数据中提取维度
    num_samples, num_locations, num_frames, num_planes, _ = data.shape
    
    # 为每架飞机的经纬度和高度创建键
    for i in range(num_planes):
        allData[f'Longitude{i+1}'] = data[:, 0, :, i, 0]  # 经度
        allData[f'Latitude{i+1}'] = data[:, 1, :, i, 0]   # 纬度
        allData[f'Altitude{i+1}'] = data[:, 2, :, i, 0]   # 高度
    
    return allData

data_path = "train_data_plane_301Frames.npy"
label_path = "train_label_plane_301Frames.pkl"


data = np.load(data_path, mmap_mode='r')

print(data.shape)

alldata = convert_data_to_dict(data)

print()

