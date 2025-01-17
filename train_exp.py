import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = '/mnt/RTdetr/RTDETR-main/dataset/dataset_visdrone/data.yaml'
    # yaml_path = '/mnt/RTdetr/RTDETR-main/dataset/vaste/data.yaml'
    check_path(yaml_path)
    model = RTDETR('/mnt/RTdetr/SO_DETR/ultralytics/cfg/models/A-Test-M-R18.yaml')
    # model = RTDETR('/mnt/RTdetr/RTDETR-main/ultralytics/cfg/models/rt-detr/A-Test-M-EV2.yaml')
    model.train(data=str(yaml_path),
                cache=False,
                imgsz=640,
                epochs=350,
                batch=4,
                workers=24,
                device='0',
                # resume='', # last.pt path
                project='runs/train/A',
                name='exp_so_R18',
                patience = 40,
                )
    print('/mnt/RTdetr/RTDETR-main/ultralytics/cfg/models/rt-detr/exp_so_r18.yaml')