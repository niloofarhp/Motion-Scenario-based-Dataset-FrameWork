import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('../gmflow')

from utils.utils import InputPadder
from gmflow.geometry import forward_backward_consistency_check
from gmflow.gmflow import GMFlow


class PredictGMFlow() :
    def __init__(self):
        parser = argparse.ArgumentParser()
        # dataset
        parser.add_argument('--padding_factor', default=16, type=int,
                            help='the input should be divisible by padding_factor, otherwise do padding')
        # resume pretrained model 
        parser.add_argument('--resume', default='../gmflow/pretrained/gmflow_sintel-0c07dcb3.pth', type=str,
                            help='resume from pretrain model for finetuing or resume from terminated training')
        parser.add_argument('--strict_resume', action='store_true')
        parser.add_argument('--no_resume_optimizer', action='store_true')

        # should not be needed as for training/distributed running
        parser.add_argument('--seed', default=326, type=int)
        parser.add_argument('--lr', default=4e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--local_rank', default=0, type=int)

        # GMFlow model
        parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
        parser.add_argument('--feature_channels', default=128, type=int)
        parser.add_argument('--upsample_factor', default=8, type=int)
        parser.add_argument('--num_transformer_layers', default=6, type=int)
        parser.add_argument('--num_head', default=1, type=int)
        parser.add_argument('--attention_type', default='swin', type=str)
        parser.add_argument('--ffn_dim_expansion', default=4, type=int)

        parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
        parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
        parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

        # evaluation
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--save_eval_to_file', action='store_true')
        parser.add_argument('--evaluate_matched_unmatched', action='store_true')

        # inference on a directory
        parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
        parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
        parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

        # self.args = parser.parse_args()
        self.args, _ = parser.parse_known_args()
        self.model = self.prepare_model()


    @torch.no_grad()
    def prepare_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = GMFlow(feature_channels=self.args.feature_channels,
                   num_scales=self.args.num_scales,
                   upsample_factor=self.args.upsample_factor,
                   num_head=self.args.num_head,
                   attention_type=self.args.attention_type,
                   ffn_dim_expansion=self.args.ffn_dim_expansion,
                   num_transformer_layers=self.args.num_transformer_layers,
                ).to(device)
    
        seed = self.args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr,
                                  weight_decay=self.args.weight_decay)

        start_epoch = 0
        start_step = 0
        # resume checkpoints
        if self.args.resume:
            print('Load checkpoint: %s' % self.args.resume)

            loc = 'cuda:{}'.format(self.args.local_rank)
            checkpoint = torch.load(self.args.resume, map_location=loc)

            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

            model.load_state_dict(weights, strict=self.args.strict_resume)

            if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                    self.args.no_resume_optimizer:
                print('Load optimizer')
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                start_step = checkpoint['step']

            print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

        model.eval()
        return model        


    @torch.no_grad()
    def predict(self,
              image1 : np.ndarray, 
              image2 : np.ndarray
    ) :
        """ prediction based on two input images """

        # Sanity check for parameter settings
        if self.args.fwd_bwd_consistency_check:
            assert self.args.pred_bidir_flow

        # should we expect torch input instead?
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        if self.args.inference_size is None:
            padder = InputPadder(image1.shape, padding_factor=self.args.padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        else:
            image1, image2 = image1[None].cuda(), image2[None].cuda()
            assert isinstance(self.args.inference_size, list) or isinstance(self.args.inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=self.args.inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=self.args.inference_size, mode='bilinear',
                                align_corners=True)

        results_dict = self.model(image1, image2,
                        attn_splits_list=self.args.attn_splits_list,
                        corr_radius_list=self.args.corr_radius_list,
                        prop_radius_list=self.args.prop_radius_list,
                        pred_bidir_flow=self.args.pred_bidir_flow,
                    )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if self.args.inference_size is None:
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        else:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / self.args.inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / self.args.inference_size[-2]
            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        # also predict backward flow
        fwd_occ = None
        bwd_occ = None

        if self.args.pred_bidir_flow:
            assert flow_pr.size(0) == 2  # [2, H, W, 2]

            if self.args.inference_size is None:
                flow_bwd = padder.unpad(flow_pr[1]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
            else:
                flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

            # forward-backward consistency check
            # occlusion is 1
            if self.args.fwd_bwd_consistency_check:
                if self.args.inference_size is None:
                    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
                    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
                else:
                    fwd_flow = flow_pr[0].unsqueeze(0)
                    bwd_flow = flow_pr[1].unsqueeze(0)

                fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)  # [1, H, W] float

                fwd_occ = fwd_occ[0].cpu().numpy()
                bwd_occ = bwd_occ[0].cpu().numpy()
        else:
            flow_bwd = None

        return flow, flow_bwd, fwd_occ, bwd_occ