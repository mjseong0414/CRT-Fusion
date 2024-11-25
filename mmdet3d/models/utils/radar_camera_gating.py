import torch.nn as nn
from mmcv.runner import auto_fp16


class RadarCameraGating(nn.Module):
    def __init__(self,
                 channel_c=80,
                 channel_r=64):
        super(RadarCameraGating, self).__init__()
        self.cam_linear = nn.Sequential(
            nn.Conv2d(channel_c, channel_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_c),
            nn.ReLU())
        self.pts_linear = nn.Sequential(
            nn.Conv2d(channel_r, channel_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_c),
            nn.ReLU())
        self.cam_atten_weight = nn.Sequential(
            nn.Conv2d(channel_c, channel_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_c),
            nn.ReLU())
        self.rad_atten_weight = nn.Sequential(
            nn.Conv2d(channel_c, channel_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_c),
            nn.ReLU())
        
    @auto_fp16(apply_to=('img_feat', 'pts_feat'))
    def forward(self, img_feat, pts_feat):
        img_feat = self.cam_linear(img_feat.float())
        pts_feat = self.pts_linear(pts_feat.float())
        fused_feat = img_feat + pts_feat
        cam_weight = self.cam_atten_weight(fused_feat).sigmoid()
        rad_weight = self.rad_atten_weight(fused_feat).sigmoid()
        fused_feat = img_feat * cam_weight + pts_feat * rad_weight

        return fused_feat
