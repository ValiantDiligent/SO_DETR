import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou, bbox_inner_iou
from ultralytics.utils.checks import check_version
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

import functools


class RTDETRLogicLoss(nn.Module):
    def __init__(self, hyp):
        super().__init__()

        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def power_transform(self, array, power=2):
        return torch.where(array < 0.5, array ** power, array ** (1/power))
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox_iou = torch.zeros(1, device=self.device)  # box iou loss
        lbox_l1 = torch.zeros(1, device=self.device)  # box l1 loss
        
        s_dec_bboxes, s_dec_scores, s_enc_bboxes, s_enc_scores, s_dn_meta = s_p
        t_dec_bboxes, t_dec_scores, t_enc_bboxes, t_enc_scores, t_dn_meta = t_p[1]
        
        s_dec_bboxes, s_dec_scores = s_dec_bboxes[:, -1], s_dec_scores[:, -1]
        t_dec_bboxes, t_dec_scores = t_dec_bboxes[:, -1], t_dec_scores[:, -1]
        
        if s_dn_meta is not None:
            _, s_dec_bboxes = torch.split(s_dec_bboxes, s_dn_meta['dn_num_split'], dim=1)
            _, s_dec_scores = torch.split(s_dec_scores, s_dn_meta['dn_num_split'], dim=1)
            s_dec_bboxes, s_dec_scores = s_dec_bboxes[-1:], s_dec_scores[-1:]
        # if t_dn_meta is not None:
        #     _, t_dec_bboxes = torch.split(t_dec_bboxes, t_dn_meta['dn_num_split'], dim=2)
        #     _, t_dec_scores = torch.split(t_dec_scores, t_dn_meta['dn_num_split'], dim=2)
        
        t_obj_scale = t_dec_scores.sigmoid().max(-1)[0].unsqueeze(-1)
        
        lbox_l1 = F.l1_loss(s_dec_bboxes, t_dec_bboxes, reduction='none') * t_obj_scale.repeat((1, 1, 4))
        # lbox_l1 = F.l1_loss(s_dec_bboxes, t_dec_bboxes, reduction='none') * self.power_transform(t_obj_scale).repeat((1, 1, 4))
        # lbox_iou = (1.0 - bbox_iou(s_dec_bboxes, t_dec_bboxes, xywh=True, GIoU=True)) * self.power_transform(t_obj_scale)
        lbox_iou = (1.0 - bbox_inner_iou(s_dec_bboxes, t_dec_bboxes, xywh=True, SIoU=True, ratio=1.25)) * self.power_transform(t_obj_scale) # Inner IoU
        lcls = nn.BCEWithLogitsLoss(reduction='none')(s_dec_scores, t_dec_scores.sigmoid()).mean(1)
        
        lbox_l1 = lbox_l1.sum() / batch['bboxes'].size(0) * 5
        lbox_iou = lbox_iou.sum() / batch['bboxes'].size(0) * 2
        lcls = lcls.sum() / (batch['bboxes'].size(0) / t_obj_scale.size(1))
        
        return lbox_l1 + lbox_iou + lcls
