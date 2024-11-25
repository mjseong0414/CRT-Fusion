import os
import copy
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.ops.nms import batched_nms
from mmdet.models import DETECTORS
from mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmdet3d.ops import bev_pool
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from torch.cuda.amp.autocast_mode import autocast
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d

from .bevdet import BEVDet
from .. import builder
import numpy as np
import time
import cv2
import pandas as pd

def generate_forward_transformation_matrix(img_meta_dict):
    res = torch.eye(3)

    if 'transformation_3d_flow' in img_meta_dict:
        for transform_type in img_meta_dict['transformation_3d_flow']:
            if transform_type == "R":
                if "pcd_rotation" in img_meta_dict:
                    res = img_meta_dict['pcd_rotation'].T @ res # .T since L158 of lidar_box3d has points @ rot
            elif transform_type == "S":
                if "pcd_scale_factor" in img_meta_dict:
                    res = res * img_meta_dict['pcd_scale_factor']
            elif transform_type == "T":
                if "pcd_trans" in img_meta_dict:
                    assert torch.tensor(img_meta_dict['pcd_trans']).abs().sum() == 0, \
                        "I'm not supporting translation rn; need to convert to hom coords which is annoying"
            elif transform_type == "HF": # Horizontal is Y apparently
                if "pcd_horizontal_flip" in img_meta_dict:
                    tmp = torch.eye(3)
                    tmp[1, 1] = -1
                    res = tmp @ res
            elif transform_type == "VF":
                if "pcd_vertical_flip" in img_meta_dict:
                    tmp = torch.eye(3)
                    tmp[0, 0] = -1
                    res = tmp @ res
            else:
                raise Exception(str(img_meta_dict))

    hom_res = torch.eye(4)
    hom_res[:3, :3] = res
    return hom_res

@DETECTORS.register_module()
class CRTFusion_light(BEVDet):
    def __init__(self, 
                 pre_process=None, 
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=1, # Number of history key frames to cat
                 history_cat_conv_out_channels=None,
                 pts_voxel_encoder=None, 
                 pts_backbone=None, 
                 pts_neck=None,
                 loss_depth_weight=3.0, 
                 loss_semantic_weight=25,
                 bev_seg_th=0.05,
                 motion_est=False,
                 motion_th=1.0,
                 **kwargs):
        super(CRTFusion_light, self).__init__(**kwargs)

        self.motion_th = motion_th
        self.single_bev_num_channels = self.img_view_transformer.numC_Trans
        self.point_cloud_range = self.pts_voxel_layer.point_cloud_range
        self.voxel_size = self.pts_voxel_layer.voxel_size
        
        if self.train_cfg is not None:
            self.out_size_factor = self.train_cfg.pts.out_size_factor
        elif self.test_cfg is not None:
            self.out_size_factor = self.test_cfg.pts.out_size_factor
        
        # Lightweight MLP
        self.embed = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True))

        # Preprocessing like BEVDet4D
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        #### Deal with history
        self.do_history = do_history
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        if self.do_history:
            self.interpolation_mode = interpolation_mode

            self.history_cat_num = history_cat_num
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)

            self.motion_gating = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels*(self.history_cat_num+1),
                          self.single_bev_num_channels*(self.history_cat_num+1), 
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(self.single_bev_num_channels*(self.history_cat_num+1)),
                nn.ReLU(True),
                nn.Sigmoid()
            )
            
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))
            
            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

            self.history_sweep_time = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            self.his_motion = None
            self.his_occ_feat = None

        self.loss_depth_weight = loss_depth_weight
        self.loss_semantic_weight = loss_semantic_weight
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        self.pts_backbone = None
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        self.pts_neck = None
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        self.motion_est = motion_est
        if self.motion_est:
            self.motion_net = nn.Sequential(
                    nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.single_bev_num_channels),
                    nn.ReLU(True),
                    nn.Conv2d(self.single_bev_num_channels, 2, kernel_size=1))
            self.occupancy_net = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(True),
                nn.Conv2d(self.single_bev_num_channels, 1, kernel_size=1),
                nn.Sigmoid())
            
        self.occ_mask_th = bev_seg_th
        self.fp16_enabled = False
    
    def reverse_bilinear(self, curr_batch_grid, his_occ_bev_feat, BCHW):
        eps = 1e-6
        B, C, H, W = BCHW
        BEV_buckets = torch.zeros(B, self.history_cat_num*C, H, W, device=curr_batch_grid.device)
        
        # Extract batch indices and BEV coordinates
        batch_indices, bev_h, bev_w = curr_batch_grid.t()
        
        h0 = torch.clamp(bev_h.floor(), 0, H - 1).long()
        w0 = torch.clamp(bev_w.floor(), 0, W - 1).long()
        h1 = torch.clamp(h0 + 1, 0, H - 1)
        w1 = torch.clamp(w0 + 1, 0, W - 1)
        
        ### Calculate interpolation weights
        distance00 = torch.sqrt((bev_h-h0)**2 + (bev_w-w0)**2)
        distance01 = torch.sqrt((bev_h-h0)**2 + (bev_w-w1)**2)
        distance10 = torch.sqrt((bev_h-h1)**2 + (bev_w-w0)**2)
        distance11 = torch.sqrt((bev_h-h1)**2 + (bev_w-w1)**2)
        
        weight00 = 1 / (distance00 + eps)
        weight01 = 1 / (distance01 + eps)
        weight10 = 1 / (distance10 + eps)
        weight11 = 1 / (distance11 + eps)
        
        total_weight = weight00 + weight01 + weight10 + weight11
        weight00 = weight00 / total_weight
        weight01 = weight01 / total_weight
        weight10 = weight10 / total_weight
        weight11 = weight11 / total_weight
        
        is_zero_distance = torch.nonzero(distance00==0)
        weight00[is_zero_distance] = 1.0
        weight01[is_zero_distance] = 0.0
        weight10[is_zero_distance] = 0.0
        weight11[is_zero_distance] = 0.0
        
        batch_indices = batch_indices.long()
        BEV_buckets[batch_indices, :, w0, h0] += weight00[:, None]*his_occ_bev_feat
        BEV_buckets[batch_indices, :, w1, h0] += weight01[:, None]*his_occ_bev_feat
        BEV_buckets[batch_indices, :, w0, h1] += weight10[:, None]*his_occ_bev_feat
        BEV_buckets[batch_indices, :, w1, h1] += weight11[:, None]*his_occ_bev_feat
        
        return BEV_buckets
    
    
    @auto_fp16()
    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size) 
        if self.pts_backbone is not None:
            x = self.pts_backbone(x)
        if self.pts_neck is not None:
            x = self.pts_neck(x)
        return x
    
    @auto_fp16()
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        neck_feats = self.img_neck(backbone_feats)
        if isinstance(neck_feats, list):
            assert len(neck_feats) == 1 # SECONDFPN returns a length-one list
            neck_feats = neck_feats[0]
            
        _, output_dim, ouput_H, output_W = neck_feats.shape
        neck_feats = neck_feats.view(B, N, output_dim, ouput_H, output_W)

        return neck_feats
    
    @force_fp32()
    def get_depth_and_segment_loss(self, depth_gt, depth, seg_gt, segment):
        """
        This was updated to be more similar to BEVDepth's original depth loss function.
        """
        B, N, H, W = depth_gt.shape
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                            self.img_view_transformer.D).to(torch.long)
        assert depth_gt.max() < self.img_view_transformer.D

        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32) # B x N x D x H x W
        depth = depth.view(B, N, self.img_view_transformer.D, H, W).softmax(dim=2)
        
        depth_gt_logit = depth_gt_logit.permute(0, 1, 3, 4, 2).view(-1, self.img_view_transformer.D)
        depth = depth.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.img_view_transformer.D)
        
        ### segmentation label and pred
        seg_gt = seg_gt.view(B*N, H, W)
        seg_labels = F.one_hot(seg_gt.long(),
                               num_classes=2).view(-1,2).float()
        segment_weight = torch.zeros_like(seg_labels[:,1:2])
        segment_weight = torch.fill_(segment_weight, 0.1)
        segment_weight[seg_labels[:,1] > 0] = 0.9
        segment = segment.view(-1, 2)
        
        depth_mask = torch.max(depth_gt_logit, dim=1).values > 0.0
        seg_labels = seg_labels[depth_mask]
        segment = segment[depth_mask]
        segment_weight = segment_weight[depth_mask]
        with autocast(enabled=False):
            loss_depth = (F.binary_cross_entropy(
                    depth[depth_mask],
                    depth_gt_logit[depth_mask],
                    reduction='none',
                )*segment_weight).sum() / max(1.0, segment_weight.sum())
            
            pred = segment
            target = seg_labels
            alpha = 0.25
            gamma = 2
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) *
                            (1 - target)) * pt.pow(gamma)
            segment_loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            segment_loss = segment_loss.sum() / max(1, len(segment_loss))
        return self.loss_depth_weight * loss_depth, self.loss_semantic_weight * segment_loss
    

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, curr_motion=None, occ_pred=None):
        device = curr_bev.device
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = torch.stack([
            generate_forward_transformation_matrix(single_img_metas) 
            for single_img_metas in img_metas], dim=0).to(curr_bev)
        curr_to_prev_lidar_rt = torch.stack([
            single_img_metas['curr_to_prev_lidar_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        B, C, H, W = curr_bev.shape
        
        # occupancy mask
        if occ_pred is not None:
            occ_mask = (occ_pred > self.occ_mask_th).permute(0,2,3,1).squeeze(-1).contiguous()
        else:
            occ_mask = torch.ones((B, H, W), device=device).bool()
        occ_bev_feat = curr_bev.permute(0,2,3,1).contiguous()[occ_mask]
        occ_grid_idx = torch.nonzero(occ_mask).float()
        
        ## Deal with first batch
        if self.his_occ_feat is None:
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()
            self.his_occ_feat = occ_bev_feat.clone()
            self.his_occ_mask = occ_mask.clone()
            if curr_motion is not None:
                self.his_motion = curr_motion.clone()
            
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)
            
            # Repeat the first frame to be history
            self.his_occ_feat = occ_bev_feat.repeat(1, self.history_cat_num)
        
        self.his_occ_feat = self.his_occ_feat.detach()
        if curr_motion is not None:
            self.his_motion = self.his_motion.detach()
        # 각 배치별 True의 개수 계산
        true_counts = self.his_occ_mask.sum(dim=(1, 2))
        for batch_idx in range(B):
            if start_of_sequence[batch_idx]:
                # Identify indices for the current batch in occ_bev_feat and batch_grid
                batch_indices = (occ_grid_idx[:, 0] == batch_idx).nonzero(as_tuple=True)[0]
                
                # Select the features and grid indices for the current batch
                batch_occ_bev_feat = occ_bev_feat[batch_indices]

                # Repeat the features and grid indices according to your requirement
                repeated_occ_bev_feat = batch_occ_bev_feat.repeat(1, self.history_cat_num)
                
                # 해당 배치의 데이터 삭제 및 삽입
                start = sum(true_counts[:batch_idx])
                end = start + true_counts[batch_idx]

                self.his_occ_feat = torch.cat([self.his_occ_feat[:start], 
                                            repeated_occ_bev_feat, 
                                            self.his_occ_feat[end:]])
                true_counts[batch_idx] = repeated_occ_bev_feat.size(0)
        
        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        
        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
        self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
        self.his_occ_mask[start_of_sequence] = occ_mask[start_of_sequence]
        if curr_motion is not None:
            self.his_motion[start_of_sequence] = 0 # 첫 sequence에서는 속도 보상을 하면 안됨

        # Generate grid
        xs = torch.linspace(0, W - 1, W, dtype=curr_bev.dtype, device=curr_bev.device).view(1, W).expand(H, W)
        ys = torch.linspace(0, H - 1, H, dtype=curr_bev.dtype, device=curr_bev.device).view(H, 1).expand(H, W)
        grid = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, H, W, 4).expand(B, H, W, 4).view(B,H,W,4,1)
        
        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=torch.float32).to(device)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        ## prev2current ego compensation matrix
        # backward of prev augmentations, prev lidar to curr lidar, forward of current augmentations,
        # transform to current grid locations
        rt_flow_p2c = (torch.inverse(feat2bev) @ forward_augs @ torch.inverse(curr_to_prev_lidar_rt)
                   @ torch.inverse(self.history_forward_augs) @ feat2bev)
        
        batch_indices = torch.nonzero(self.his_occ_mask)[..., 0]
        grid = grid[self.his_occ_mask]
        if curr_motion is not None:
            his_occ_motion = self.his_motion.permute(0,2,3,1).contiguous()[self.his_occ_mask]
            moving_occ_mask = torch.sqrt((his_occ_motion[:,0]**2)+(his_occ_motion[:,1]**2)) > self.motion_th
            his_occ_motion = his_occ_motion / self.voxel_size[0] / self.out_size_factor
            his_occ_motion = his_occ_motion * self.history_cam_sweep_freq
            grid[moving_occ_mask][:, 0:2] = grid[moving_occ_mask][:, 0:2] + his_occ_motion[moving_occ_mask].unsqueeze(-1)
        
        selected_transforms = torch.index_select(rt_flow_p2c, 0, batch_indices.to(device))
        grid = torch.bmm(selected_transforms, grid)
        grid = torch.cat((batch_indices.unsqueeze(-1),grid[:,:2,0].reshape(-1,2)), dim=1)
        prev2curr_bev = self.reverse_bilinear(grid, self.his_occ_feat, curr_bev.shape)
        
        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)
        
        feats_cat = torch.cat([curr_bev, prev2curr_bev], dim=1) # B x (1 + T) * 80 x H x W
        
        motion_weight = self.motion_gating(feats_cat)
        feats_cat = feats_cat * motion_weight
        
        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
            feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3]) # B x (1 + T) x 80 x H x W
        feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W
        
        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 x H x W
        
        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x C x H x W
        
        # Update history by moving everything down one group of single_bev_num_channels channels
        self.his_occ_feat = prev2curr_bev.permute(0,2,3,1)[occ_mask][:, :-self.single_bev_num_channels].detach().clone()
        self.his_occ_feat = torch.cat((occ_bev_feat.detach().clone(), self.his_occ_feat),dim=1)
        self.his_occ_mask = occ_mask.detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if curr_motion is not None:
            self.his_motion = curr_motion.detach().clone()
        return feats_to_return.clone()

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        return x
    
    def extract_img_feat(self, radar, img, img_metas, gt_bboxes_3d=None):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        imgs = inputs[0].view(B, N, 1, 3, H, W)
        imgs = torch.split(imgs, 1, dim=2)
        imgs = [tmp.squeeze(2) for tmp in imgs] # List of imgs each B x N x 3 x H x W
  
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]

        extra = [rots.view(B, 1, N, 3, 3),
                 trans.view(B, 1, N, 3),
                 intrins.view(B, 1, N, 3, 3),
                 post_rots.view(B, 1, N, 3, 3),
                 post_trans.view(B, 1, N, 3)]
        extra = [torch.split(t, 1, dim=1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra] # each B x N x 3 (x 3)
        rots, trans, intrins, post_rots, post_trans = extra
        
        curr_img_encoder_feats = self.image_encoder(imgs[0])
        radar_bev_feat = self.extract_pts_feat(radar)
        lidar2ego = []
        for b in range(B):
            lidar2ego.append(img_metas[b]['curr_lidar_to_ego_rt'])
        lidar2ego = torch.tensor(np.stack(lidar2ego), device=radar_bev_feat.device)
        fused_bev_feat, depth_digit, seg_prob = self.img_view_transformer(
                                            curr_img_encoder_feats, radar_bev_feat,
                                            rots[0], trans[0], intrins[0], post_rots[0], 
                                            post_trans[0], lidar2ego)
        fused_bev_feat = self.pre_process_net(fused_bev_feat)[0] # singleton list
        fused_bev_feat = self.embed(fused_bev_feat) 
        
        if self.motion_est:
            curr_motion = self.motion_net(fused_bev_feat)
            occ_pred = self.occupancy_net(fused_bev_feat)
        
        # Fuse History
        if self.do_history:
            if self.motion_est:
                fused_bev_feat = self.fuse_history(fused_bev_feat, img_metas, curr_motion=curr_motion, occ_pred=occ_pred)
            else:
                fused_bev_feat = self.fuse_history(fused_bev_feat, img_metas)
        
        x = self.bev_encoder(fused_bev_feat)
        
        if self.motion_est:
            return x, depth_digit, seg_prob, curr_motion, occ_pred
        else:
            return x, depth_digit, seg_prob

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      radar=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        losses = dict()
        if self.motion_est:
            img_feats, depth_digit, seg_prob, curr_motion, occ_pred = self.extract_img_feat(radar, img_inputs, img_metas, gt_bboxes_3d)
        else:
            img_feats, depth_digit, seg_prob = self.extract_img_feat(radar, img_inputs, img_metas, gt_bboxes_3d)
        
        # If we're training depth and seg...
        depth_gt = img_inputs[-2]
        seg_gt = img_inputs[-1]
        loss_depth, loss_seg = self.get_depth_and_segment_loss(depth_gt, depth_digit, seg_gt, seg_prob)
        losses['loss_depth'] = loss_depth
        losses['loss_seg'] = loss_seg
        
        # Get box losses
        bbox_outs = self.pts_bbox_head(img_feats)
        if self.motion_est:
            losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs, motion_est=curr_motion, occ_pred=occ_pred, sample_index=img_metas[0]['sample_index'])
        else:
            losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs)
        losses.update(losses_pts)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, radar=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if num_augs==1:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], radar[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img_inputs, radar, **kwargs)
    
    def simple_test(self, points, img_metas, img=None, radar=None, rescale=False, **kwargs):
        if self.motion_est:
            img_feats, _, _, _, _ = self.extract_img_feat(radar, img, img_metas)
        else:
            img_feats, _, _ = self.extract_img_feat(radar, img, img_metas)
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        bbox_list = [dict(pts_bbox=bbox_pts[0])]

        return bbox_list

    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def aug_test(self, points, img_metass, imgs=None, radars=None, rescale=False):
        """Test function with augmentaiton."""
        aug_bboxes = []
        device = imgs[0][0].device
        for img_metas, img, radar in zip(img_metass, imgs, radars):
            if self.motion_est:
                img_feats, _, _, _, _ = self.extract_img_feat(radar, img, img_metas)
            else:
                img_feats, _, _ = self.extract_img_feat(radar, img, img_metas)
            outs = self.pts_bbox_head(img_feats)
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_results[0])
        
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metass,
                                            self.pts_bbox_head.test_cfg,
                                            device)
        return [merged_bboxes]