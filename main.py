#!/usr/bin/env python
import argparse
import sys
import warnings
import torch
num_devices = torch.cuda.device_count()
for i in range(num_devices):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

print("CUDA available:", torch.cuda.is_available())  # 应该返回 True
print("CUDA device count:", torch.cuda.device_count())  # 应该 >=

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook when the forward contains multiple autograd Nodes")

# 你的其他代码..
# torchlight
# import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    processors['demo_skeleton_explainer'] = import_class('processor.demo_skeleton_explainer.DomeSkeletonExplainer')
    processors['demo_skeleton_cam'] = import_class('processor.demo_skeleton_cam.DemoSkeletonCam')
    processors['GNNExplainerwithcff'] = import_class('processor.GNNExplainerwithcff.Explainer')
    processors['gnn_explainer'] = import_class('processor.gnn_explainer.GNNExplainer')
    processors['gnn_explainer_oral'] = import_class('processor.gnn_explainer_oral.GNNExplainer')
    processors['JustExplainer'] = import_class('processor.JustExplainer.Explainer')
    processors['JustExplainer_oral'] = import_class('processor.JustExplainer_oral.Explainer')
    processors['ShapleyCam'] = import_class('processor.ShapleyCam.Explainer')
    processors['demo_skeleton_cam_1'] = import_class('processor.demo_skeleton_cam_1.DemoSkeletonCam')
    processors['our'] = import_class('processor.our.Explainer')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    # arg.processor = 'our'
    # arg.processor = 'demo_skeleton_cam'
    if not arg.processor:
        arg.processor = 'our'

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])


    p.start()
