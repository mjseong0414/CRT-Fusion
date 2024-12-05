import os
import copy
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import NECKS
from mmdet3d.ops import bev_pool
from .. import builder
from .view_transformer import ViewTransformerLiftSplatShoot, SELikeModule
from mmdet3d.models.utils import PointGating, RadarCameraGating


@NECKS.register_module()
class ViewTransformerCRTFusion(ViewTransformerLiftSplatShoot):
    def __init__(self, 
                 extra_depth_net,
                 loss_depth_weight, 
                 se_config=dict(), 
                 pv_seg_th=0.25,
                 radar_selection=128,
                 light_version=False,
                 **kwargs):
        super(ViewTransformerCRTFusion, self).__init__(**kwargs)
        self.radar_selection = radar_selection
        self.loss_depth_weight = loss_depth_weight
        self.extra_depthnet = builder.build_backbone(extra_depth_net)
        self.featnet = nn.Conv2d(self.numC_input,
                                 self.numC_Trans,
                                 kernel_size=1,
                                 padding=0)
        self.depthnet = nn.Conv2d(extra_depth_net['num_channels'][0],
                                  self.D+2,
                                  kernel_size=1,
                                  padding=0)
        self.se = SELikeModule(self.numC_input,
                               feat_channel=extra_depth_net['num_channels'][0],
                               **se_config)
        self.fc_h = nn.Sequential(nn.Conv1d(self.numC_input, self.numC_input//2, kernel_size=1),
                                 nn.BatchNorm1d(self.numC_input//2),
                                 nn.ReLU())
        self.fc_w = nn.Sequential(nn.Conv1d(self.numC_input, self.numC_input, kernel_size=1),
                                 nn.BatchNorm1d(self.numC_input),
                                 nn.ReLU())
        self.cam_feat_weight = nn.Sequential(
            nn.Conv2d(self.numC_input, self.numC_input, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.numC_input),
            nn.ReLU())
        self.rad_feat_weight = nn.Sequential(
            nn.Conv2d(self.numC_input, self.numC_input, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.numC_input),
            nn.ReLU())
        self.encoded_bev = nn.Sequential(
            nn.Conv2d(self.numC_Trans+1, self.numC_Trans, kernel_size=1),
            nn.BatchNorm2d(self.numC_Trans),
            nn.ReLU())
        
        self.PointGating = PointGating()
        if not light_version:
            self.fcv2 = nn.Sequential(nn.Conv1d(self.numC_Trans, self.numC_Trans, kernel_size=1),
                                    nn.BatchNorm1d(self.numC_Trans),
                                    nn.ReLU())
            self.rcgating = RadarCameraGating(channel_r=80)
            
        else:
            self.fcv2 = nn.Sequential(nn.Conv1d(64, self.numC_Trans, kernel_size=1),
                                    nn.BatchNorm1d(self.numC_Trans),
                                    nn.ReLU())
            self.rcgating = RadarCameraGating()
        self.depth_threshold = 1/self.D
        self.pv_seg_th = pv_seg_th
        self.fp16_enabled = False

    def RCA_Attention(self, lidar2ego, radar_bev_feat, img_feats, geom):
        B,C,H,W = radar_bev_feat.shape
        _,N,Ci,Hi,Wi = img_feats.shape
        device = radar_bev_feat.device
        img_feats_flip = torch.flip(img_feats, [3])
        
        # Generate grid
        xs = torch.linspace(0, W - 1, W, dtype=radar_bev_feat.dtype, device=radar_bev_feat.device).view(1, W).expand(H, W)
        ys = torch.linspace(0, H - 1, H, dtype=radar_bev_feat.dtype, device=radar_bev_feat.device).view(H, 1).expand(H, W)
        grid = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, H, W, 4).expand(B, H, W, 4).view(B,H,W,4,1)
        
        feat2bev = torch.zeros((4,4),dtype=torch.float32).to(device)
        feat2bev[0, 0] = self.dx[0]
        feat2bev[1, 1] = self.dx[1]
        feat2bev[0, 3] = self.bx[0] - self.dx[0] / 2.
        feat2bev[1, 3] = self.bx[1] - self.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        
        rt_flow = (torch.inverse(feat2bev) @ lidar2ego.float() @ feat2bev)
        grid_ = rt_flow.view(B, 1, 1, 4, 4) @ grid
        # normalize and sample
        normalize_factor = torch.tensor([W - 1.0, H - 1.0], dtype=radar_bev_feat.dtype, device=radar_bev_feat.device)
        grid__ = grid_[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        radar_bev_feat_ego = F.grid_sample(radar_bev_feat, grid__.to(radar_bev_feat.dtype), align_corners=True, mode='bilinear')
        
        geom_ = geom[:, :, self.D//2,(img_feats.shape[3]//2 - 1),:].contiguous().view(-1, 3)
        geom_angle = torch.atan2(geom_[:,1],geom_[:,0]).view(B, -1).unsqueeze(2)
        
        BEV_grid_ego = (feat2bev.repeat(B,1,1).view(B,1,1,4,4) @ grid_).squeeze(-1)[..., :2]
        BEV_grid_angle = torch.atan2(BEV_grid_ego[..., 1], BEV_grid_ego[...,0]).view(B, H*W)
        differences = torch.abs(BEV_grid_angle.unsqueeze(1) - geom_angle)
        
        sorted_indices = torch.argsort(differences, dim=2)
        selected_indices = sorted_indices[..., :self.radar_selection]
        
        expanded_indices = selected_indices.unsqueeze(1).expand(-1, C, -1, -1).reshape(B, C, 6*Wi*self.radar_selection)
        BEV_features_flattened = radar_bev_feat_ego.view(B,C,H*W)
        selected_features = torch.gather(BEV_features_flattened, 2, expanded_indices).view(B, C, 6*Wi, self.radar_selection)
        selected_features = selected_features.permute(0, 2, 3, 1).contiguous().view(-1, self.radar_selection, C)
        
        selected_tmp_point_enc = self.fcv2(selected_features.permute(0,2,1))
        selected_pts = selected_tmp_point_enc.permute(0,2,1).contiguous()
        
        # img height collapse
        img_feats_clone = img_feats_flip.clone()
        img_feat_collapsed_h = img_feats_clone.permute(0,2,3,1,4)
        img_feat_collapsed_h = img_feat_collapsed_h.reshape(B, Ci, Hi, N*Wi).max(2).values
        img_feat_collapsed_h = self.fc_h(img_feat_collapsed_h).permute(0, 2, 1)
        
        # img width collapse
        img_feat_collapsed_w = img_feats_clone.permute(0,2,4,1,3)
        img_feat_collapsed_w = img_feat_collapsed_w.reshape(B, Ci, Wi, N*Hi).max(2).values
        img_feat_collapsed_w = self.fc_w(img_feat_collapsed_w).view(B, -1, N, Hi).permute(0,2,1,3).unsqueeze(4).contiguous()
        
        selected_pts = torch.cat((selected_pts,img_feat_collapsed_h.reshape(-1, 1, 256).repeat(1,self.radar_selection,1)), dim=2)
        img_radar_attn = self.PointGating(selected_pts)
        img_radar_attn = img_radar_attn.view(B, N, Wi, -1).permute(0,1,3,2)
        
        img_radar_attn = (img_radar_attn.unsqueeze(3) + img_feat_collapsed_w).contiguous().view(B*N, -1, Hi, Wi)
        img_feat = img_feats.view(B*N, Ci, Hi, Wi)
        
        fused_feat = img_feat + img_radar_attn
        cam_weight = self.cam_feat_weight(fused_feat).sigmoid()
        rad_weight = self.rad_feat_weight(fused_feat).sigmoid()
        results = (img_feat * cam_weight + img_radar_attn * rad_weight)
        return results.view(B,N,Ci,Hi,Wi)
    
    def voxel_pooling_crtfusion(self, geom_feats, x, kept, depth_prob=None):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        
        # filter out points that are outside box
        kept = kept.view(Nprime)
        kept &= (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        if self.max_drop_point_rate > 0.0 and self.training:
            drop_point_rate = torch.rand(1)*self.max_drop_point_rate
            kept = torch.rand(x.shape[0])>drop_point_rate
            x, geom_feats = x[kept], geom_feats[kept]

        if self.use_bev_pool:
            try:
                final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0],
                                   self.nx[1])
            except:
                breakpoint()
            final = final.transpose(dim0=-2, dim1=-1)
        else:
            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

            # cumsum trick
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        
        if depth_prob != None:
            # 'depth_prob' is in ego vehicle coordinates. So we should transform ego2LiDAR coordinates
            from torch_scatter import scatter_max
            ### bev_occ_prob computation ###
            depth_prob = depth_prob.reshape(Nprime)
            depth_prob = depth_prob[kept]
            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            depth_prob, geom_feats, ranks = depth_prob[sorts], geom_feats[sorts], ranks[sorts]
            
            unique_ranks, inverse_indices, counts = torch.unique_consecutive(ranks, return_inverse=True, return_counts=True)
            first_index = torch.cumsum(counts, dim=0) - counts
            geom_feats_first = geom_feats[first_index]
            depth_max_values, _ = scatter_max(depth_prob, ranks.long(), dim=0, dim_size=unique_ranks.max() + 1)
            depth_max_for_unique_ranks = depth_max_values[unique_ranks.long()]
            
            # griddify (B x 1 x X x Y)
            bev_occ_prob = torch.zeros((B, self.nx[1].int().item(), self.nx[0].int().item()), device=x.device, dtype=torch.float32)
            bev_occ_prob[geom_feats_first[:, 3], geom_feats_first[:, 1], geom_feats_first[:, 0]] = depth_max_for_unique_ranks
        
            return final, bev_occ_prob.unsqueeze(1)
        
        return final
    
    def get_depth_seg_softmax(self, depth_digit, seg_digit):
        depth = depth_digit.softmax(dim=1)
        seg = seg_digit.softmax(dim=1)
        return depth, seg
    
    @auto_fp16(apply_to=('curr_sem_feats', ))
    def get_mono_depth(self, curr_sem_feats, rots, trans, intrins, post_rots, post_trans):
        B, N, sem_C, sem_H, sem_W = curr_sem_feats.shape
        curr_sem_feats = curr_sem_feats.view(B * N, sem_C, sem_H, sem_W)
        curr_img_feat = self.featnet(curr_sem_feats)
        mono_depth_feat = curr_sem_feats
        cam_params = torch.cat([intrins.reshape(B*N,-1),
                               post_rots.reshape(B*N,-1),
                               post_trans.reshape(B*N,-1),
                               rots.reshape(B*N,-1),
                               trans.reshape(B*N,-1)],dim=1)
        mono_depth_feat = self.se(mono_depth_feat, cam_params)
        mono_depth_feat = self.extra_depthnet(mono_depth_feat)[0]
        mono_depth_digit = self.depthnet(mono_depth_feat)

        return mono_depth_digit, curr_img_feat

    @auto_fp16(apply_to=('curr_sem_feats','radar_bev_feat'))
    def forward(self, 
                curr_sem_feats, radar_bev_feat,
                rots, trans, intrins, post_rots, post_trans, lidar2ego):
        B, N, sem_C, sem_H, sem_W = curr_sem_feats.shape
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        curr_semdepth_feats = self.RCA_Attention(lidar2ego, radar_bev_feat, curr_sem_feats, geom)
        mono_depth_seg_digit, curr_img_feat = self.get_mono_depth(curr_semdepth_feats, rots, trans, intrins, post_rots, post_trans)
        depth_digit = mono_depth_seg_digit[:, :self.D]
        seg_digit = mono_depth_seg_digit[:, self.D:self.D+2]
            
        depth_prob, seg_prob = self.get_depth_seg_softmax(depth_digit, seg_digit)
        kept = (depth_prob >= self.depth_threshold) * (seg_prob[:,1:2] >= self.pv_seg_th)

        ### Lift
        volume = depth_prob.unsqueeze(1) * curr_img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, sem_H, sem_W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)
        
        ### Splat
        bev_feat, bev_occ_prob = self.voxel_pooling_crtfusion(geom, volume, kept, depth_prob=depth_prob)
        bev_occ_prob = (bev_occ_prob != 0)*1
        fused_bev_feat = self.rcgating(bev_feat, radar_bev_feat)
        bev_for_occ_pred = torch.cat((fused_bev_feat, bev_occ_prob), dim=1)
        fused_bev_feat = self.encoded_bev(bev_for_occ_pred)
        return fused_bev_feat, depth_digit[:, :self.D], seg_prob
