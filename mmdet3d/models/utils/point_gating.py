import torch
import torch.nn as nn
from mmcv.runner import auto_fp16

class PointGating(nn.Module):
    def __init__(self, channels=336, channels2=512):
        super().__init__()
        self.channels = channels
        self.channels2 = channels2
        self.pts_encode = nn.Sequential(
            nn.Conv1d(channels, self.channels, kernel_size=1),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
            nn.Conv1d(channels, self.channels2, kernel_size=1),
            nn.BatchNorm1d(self.channels2),
            nn.ReLU())
        self.pts_attn_weight = nn.Sequential(
            nn.Conv1d(self.channels2, 1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.ReLU())
    
    def forward(self, points) -> torch.Tensor:
        points = points.permute(0,2,1).contiguous()
        enc_points = self.pts_encode(points)
        results = (enc_points * self.pts_attn_weight(enc_points).softmax(-1))
        results = results.sum(2) 
        
        return results