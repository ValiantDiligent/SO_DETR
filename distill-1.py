import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.distill import RTDETRDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/mnt/RTdetr/RTDETR-main/ultralytics/cfg/models/rt-detr/A-Test-M-EV2.yaml',
        'data':'/mnt/RTdetr/RTDETR-main/dataset/dataset_visdrone/data.yaml',
        'imgsz': 640,
        'epochs': 600,
        'patience': 40,
        'batch': 8,
        'workers': 8,
        'cache': True,
        'device': '0',
        'project':'runs/distill',
        'name':'rtdetr-logics-exp1',
        
        # distill
        'prune_model': False,
        'teacher_weights': '/mnt/RTdetr/RTDETR-main/runs/train/ijcnn/exp_so_r50/weights/best.pt',
        'teacher_cfg': '/mnt/RTdetr/RTDETR-main/ultralytics/cfg/models/rt-detr/A-Test-r50-M.yaml',
        'kd_loss_type': 'logical',
        'kd_loss_decay': 'constant',
        'kd_loss_epoch': 1.0,
        
        'logical_loss_type': 'logical',
        'logical_loss_ratio': 0.1,
        
    }
    
    model = RTDETRDistiller(overrides=param_dict)
    model.distill()