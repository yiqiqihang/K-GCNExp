#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np
import torch
import skvideo.io
import cv2

from .io import IO
import tools
import tools.utils as utils
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new

import cv2

max_body = 2
num_joint = 25
max_frame = 300

class DemoSkeletonOffline(IO):

    def start(self):
        
        '''
        # initiate
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # pose estimation
        video, data_numpy = self.pose_estimation()
        '''

        # initiate
        # label from http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
        label_name_path = './resource/ntu_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # skeleton feature extract

        skeleton_name = self.arg.skeleton.split('/')[-1].split('.')[0]

        output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_name)
        output_ske_video_path = '{}/ske_video/{}.mp4'.format(output_result_dir, skeleton_name)
        output_ske_img_path = '{}/ske_image/{}'.format(output_result_dir, skeleton_name)
        output_corr_video_path = '{}/corr_video/{}.mp4'.format(output_result_dir, skeleton_name)
        output_corr_img_path = '{}/corr_image/{}'.format(output_result_dir, skeleton_name)

        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4])
        subject_id = int(
            skeleton_name[skeleton_name.find('P') + 1:skeleton_name.find('P') + 4])
        camera_id = int(
            skeleton_name[skeleton_name.find('C') + 1:skeleton_name.find('C') + 4])

        data_numpy = read_xyz_new(
            self.arg.skeleton, max_body=max_body, num_joint=num_joint)
        # fp[:, 0:data.shape[1], :, :] = data

        # action recognition
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

        # model predict
        voting_label_name, voting_proba, \
            video_label_name, video_proba,output, \
                intensity, hidden_feature = \
            self.extract_feature_and_probability(data)
        # voting_label_name是指整个骨架视频序列的最终分类，并转换成单词向量
        # video_label_name是指每一帧的骨架的分类
        # output是指每一帧、每个人、每个节点的分类，是一个多维向量
        # intensity是指每个关节的特征强度的L2范数
        # 修改后，feature 指每一层的feature的list, intensity指的是每一层的feature的L2范数的list

        # render the video
        # images = self.render_video(data_numpy, voting_label_name,
        #                     video_label_name, intensity, video)

        images = self.render_skeleton(data_numpy, voting_label_name,voting_proba,
                             video_label_name, video_proba,intensity)

        # 保存骨架可视化图像
        print('\nSaving...')
        if not os.path.exists('{}/ske_video/'.format(output_result_dir)):
            os.makedirs('{}/ske_video/'.format(output_result_dir))
        if not os.path.exists('{}/ske_image/'.format(output_result_dir)):
            os.makedirs('{}/ske_image/'.format(output_result_dir))
        writer = skvideo.io.FFmpegWriter(output_ske_video_path,
                                        outputdict={'-b': '300000000'})
        num = 1
        for img in images:
            writer.writeFrame(img)
            img_name = '{}_{}.png'.format(output_ske_img_path, num)
            num += 1
            cv2.imwrite(img_name,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        writer.close()
        print('The Demo video has been saved in {}.'.format(output_ske_video_path))
        print('The Demo image has been saved in {}.'.format(output_ske_img_path))
        
        # 保存相关性图可视化图像
        corr_img = self.render_correlation(hidden_feature)

        if not os.path.exists('{}/corr_video/'.format(output_result_dir)):
            os.makedirs('{}/corr_video/'.format(output_result_dir))
        if not os.path.exists('{}/corr_image/'.format(output_result_dir)):
            os.makedirs('{}/corr_image/'.format(output_result_dir))
        writer = skvideo.io.FFmpegWriter(output_corr_video_path,
                                        outputdict={'-b': '300000000'})
        num = 1
        for img in corr_img:
            writer.writeFrame(img)
            img_name = '{}_{}.png'.format(output_corr_img_path, num)
            num += 1
            cv2.imwrite(img_name,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.close()
        print('The Demo video has been saved in {}.'.format(output_corr_video_path))
        print('The Demo image has been saved in {}.'.format(output_corr_img_path))


    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        print("output shape", output.shape)
        print("feature shape", feature.shape)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def extract_feature_and_probability(self, data):
        # forward
        output, feature = self.model.extract_hidden_feature(data)
        
        output = output[0]
        # feature 维度为(N, c, t, v, M), 其中N为1
        feature = [f[0] for f in feature]
        intensity = [(f*f).sum(dim=0)**0.5 for f in feature]
        intensity = [i.cpu().detach().numpy() for i in intensity]

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_proba = output.sum(dim=3).sum(
            dim=2).sum(dim=1).softmax(dim=0).max().detach().numpy()
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        video_proba = list()
        for t in range(num_frame):
            frame_label_name = list()
            frame_proba = list()
            for m in range(num_person):
                person_proba = output[:, t, :, m].sum(dim=1).softmax(dim=0).max().detach().numpy()
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_proba.append(person_proba)
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
            video_proba.append(frame_proba)
        return voting_label_name, voting_proba, \
            video_label_name, video_proba, \
            output, intensity, feature

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def render_skeleton(self, data_numpy, voting_label_name, voting_proba, video_label_name, video_proba, intensity):
        images = utils.visualization_skeleton.stgcn_visualize_hidden(
            data_numpy,
            self.model.graph.edge,
            intensity,
            voting_label_name,
            voting_proba,
            video_label_name,
            video_proba,
            self.arg.height)
        return images

    def render_correlation(self, hidden_feature):
        return utils.visualization_skeleton.extract_correlation(hidden_feature)

    '''
    def pose_estimation(self):
        # load openpose python api
        if self.arg.openpose is not None:
            sys.path.append('{}/python'.format(self.arg.openpose))
            sys.path.append('{}/build/python'.format(self.arg.openpose))
        try:
            from openpose import pyopenpose as op
        except:
            print('Can not find Openpose Python API.')
            return


        video_name = self.arg.video.split('/')[-1].split('.')[0]

        # initiate
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO')
        opWrapper.configure(params)
        opWrapper.start()
        self.model.eval()
        video_capture = cv2.VideoCapture(self.arg.video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # pose estimation
        start_time = time.time()
        frame_index = 0
        video = list()
        while(True):

            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            video.append(orig_image)

            # pose estimation
            datum = op.Datum()
            datum.cvInputData = orig_image
            opWrapper.emplaceAndPop([datum])
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)
            if len(multi_pose.shape) != 3:
                continue

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
            multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1

            print('Pose estimation ({}/{}).'.format(frame_index, video_length))

        data_numpy = pose_tracker.get_skeleton_sequence()
        return video, data_numpy
    '''
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--skeleton',
                            default='../NTU-RGB-D-120/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/S001C001P001R001A001.skeleton',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--output_dir',
                            default='./data/demo_skeleton/result',
                            help='Path to save results')
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/ntu-xview/demo_skeleton_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
'''
class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
'''