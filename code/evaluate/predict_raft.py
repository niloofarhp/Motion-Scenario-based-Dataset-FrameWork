import argparse
import sys
import numpy as np

import torch 

# append path to raft
sys.path.append('../RAFT/core')
from raft import RAFT

from utils.utils import InputPadder

DEVICE = 'cuda'

class PredictRAFT() :
    def __init__(self):
        # --model=models/raft-things.pth
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='../RAFT/models/raft-things.pth', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        # self.args = parser.parse_args()
        self.args, _ = parser.parse_known_args()
        self.model = self.prepare_model()


    def prepare_model(self):
        model = torch.nn.DataParallel(RAFT(self.args))
        model.load_state_dict(torch.load(self.args.model))
        model = model.module
        model.to(DEVICE)
        model.eval()
        return model

    @staticmethod
    def img_to_torch( img_in
    ):
        img = torch.from_numpy(img_in).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    @torch.no_grad()
    def predict(self,
              image1 : np.ndarray, 
              image2 : np.ndarray
    ) :
        img_t1 = PredictRAFT.img_to_torch(image1)
        img_t2 = PredictRAFT.img_to_torch(image2)
        padder = InputPadder(img_t1.shape)
        img_t1, img_t2 = padder.pad(img_t1, img_t2)
        _, flow_up = self.model(img_t1, img_t2, iters=20, test_mode=True)
        flow_up_vis = flow_up[0].permute(1,2,0).cpu().numpy()
        return flow_up_vis, None, None, None