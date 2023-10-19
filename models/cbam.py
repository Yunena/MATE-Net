import torch
import torch.nn as nn

class SpatialAttn(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttn(nn.Module):
    def __init__(self,in_planes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, max(in_planes // 16,1), 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(max(in_planes // 16,1), in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()



    def forward(self,x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    

class AttnConvBlock(nn.Module):
    def __init__(self,in_planes):
        super().__init__()

        self.sa = SpatialAttn()
        self.ca = ChannelAttn(in_planes)

    def forward(self,x):
        out = x
        out = self.ca(out)*out
        out = self.sa(out)*out
        return out







