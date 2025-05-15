import sys
import argparse
import numpy as np
import torch
import os

# append path to ARFlow
cwd_path = os.getcwd()
# print("\n[predict_arflow.py] Runtime Directory:", cwd_path)

arflow_path = os.path.join(cwd_path, 'ARFlow/')

sys.path.append(arflow_path)
# print("[predict_arflow.py] System path:", sys.path)

from models.pwclite import PWCLite
from utils.flow_utils import resize_flow
from utils.torch_utils import restore_model
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms


class PredictARFlow():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', default='./ARFlow/checkpoints/KITTI15/pwclite_ar.tar')
        args, _ = parser.parse_known_args()
        cfg = {
            'model': {
                'upsample': True,
                'n_frames': 2,
                'reduce_dense': True
            },
            'pretrained_model': args.model,
            'test_shape': [256, 256],
        }
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.prepare_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def prepare_model(self):
        model = PWCLite(self.cfg.model)
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)

    def predict(self, image1: np.ndarray, image2: np.ndarray):
        imgs = [image1.astype(np.float32), image2.astype(np.float32)]
        h, w = imgs[0].shape[:2]
        flow_12 = self.run(imgs)['flows_fw'][0]
        flow_12 = resize_flow(flow_12, (h, w))
        np_flow_12_vis = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
        return np_flow_12_vis, None, None, None
