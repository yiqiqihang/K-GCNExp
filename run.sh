#!/bin/bash
echo "开始运行"
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --valid csub --cam_type scorecam > /home/mxh/woke_three/st-gcn/output/scorecam_$(date +%Y%m%d_%H%M%S).log 2>&1
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --valid csetup --cam_type scorecam > /home/mxh/woke_three/st-gcn/output/scorecam_$(date +%Y%m%d_%H%M%S).log 2>&1
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --valid xview --cam_type scorecam > /home/mxh/woke_three/st-gcn/output/scorecam_$(date +%Y%m%d_%H%M%S).log 2>&1
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --valid xsub --cam_type scorecam > /home/mxh/woke_three/st-gcn/output/scorecam_$(date +%Y%m%d_%H%M%S).log 2>&1

# gradcam √
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcam --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# gradcampp √
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcampp --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcampp --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcampp --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcampp --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcampp --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type gradcampp --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# GNNExplainer √
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type gnnexplainer --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type gnnexplainer --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type gnnexplainer --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type gnnexplainer --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type gnnexplainer --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type gnnexplainer --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# ablationcam √
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type ablation --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type ablation --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type ablation --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type ablation --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type ablation --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type ablation --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# scorecam √
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type scorecam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type scorecam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type scorecam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type scorecam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type scorecam --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type scorecam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# cff √
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type cff --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type cff --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type cff --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type cff --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type cff --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py GNNExplainerwithcff --exp_type cff --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# bicam √
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type bicam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type bicam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type bicam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type bicam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type bicam --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type bicam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# crcam √
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type imcam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type imcam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type imcam --run_device cuda:2 --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type imcam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type imcam --run_device cuda:2 --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py demo_skeleton_cam --cam_type imcam --run_device cuda:2 --data_level instance --skeleton S022C003P061R001A112  --valid csetup

# k-GCNExp
python /home/mxh/woke_three/st-gcn/main.py our --exp_type explainer  --data_level instance --skeleton S001C001P001R001A007  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py our --exp_type explainer  --data_level instance --skeleton S001C001P001R001A044  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py our --exp_type explainer  --data_level instance --skeleton S001C001P001R001A055  --valid xsub
python /home/mxh/woke_three/st-gcn/main.py our --exp_type explainer  --data_level instance --skeleton S022C003P061R001A061  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py our --exp_type explainer  --data_level instance --skeleton S029C002P080R002A100  --valid csetup
python /home/mxh/woke_three/st-gcn/main.py our --exp_type explainer  --data_level instance --skeleton S022C003P061R001A112  --valid csetup





echo "全部完成"