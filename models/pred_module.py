#extra attention module

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cbam import AttnConvBlock
from utils.resample import re_sample


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, clinical_num=0, idx = 0):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.attn = AttnConvBlock(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.idx = idx
        self.relu_ = nn.ReLU(inplace=True)

    def forward(self,x):
        #print('OK')
        y=None
        ca=None
        sa=None
        #print(type(x))
        if isinstance(x,tuple):
            #print(len(x))
            x, y, ca, sa = x
            #print(x.size(),y.size(),sa.size())


        residual = x
        #print(ca,sa)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)



        out = self.attn(out)




        if self.downsample is not None:
            residual = self.downsample(x)

        if sa is not None:
            #print(out.size())
            if out.size()[1]==residual.size()[1]:
                out+=residual
            else:
                out[:,0:512]+=residual
            return out, y, ca, sa
        else:
            out += residual
            out = self.relu_(out)
            return out




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, clinical_num=0, idx = 0):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes*(clinical_num+1), planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.idx = idx
        self.relu_ = nn.ReLU(inplace=True)

    def forward(self, x):
        ca=None
        sa=None
        if isinstance(x,tuple):
            x, ca, sa = x

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        outlist = []
        outlist.append(out)

        if ca is not None:
            cai = ca[:,self.idx]
            for i in range(len(out)):
                outlist.append(cai[i]*out[i])
        if sa is not None:
            sai = sa[:,self.idx]
            for i in range(1,len(outlist)):
                outlist[i] = sai[i]*outlist[i]

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        outlist[0] = out

        outlist = torch.stack(tuple(outlist),dim=1)


        outlist = self.relu_(outlist)
        return outlist


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=9,#2label+7CTP
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1,
                 clinical_num = 0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        #self.attn = AttnConvBlock(n_input_channels)


        self.conv1 = conv1x1x1(n_input_channels if n_input_channels>0 else 1,32,stride=(1,2,2))
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = conv1x1x1(32,self.in_planes)
        self.bn2 = nn.BatchNorm3d(self.in_planes)
        self.conv3 = conv1x1x1(self.in_planes,32)
        self.bn3 = nn.BatchNorm3d(32)

        self.attn = nn.Sequential(AttnConvBlock(in_planes=n_input_channels + clinical_num))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type,input_num=n_input_channels+clinical_num)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, clinical_num = 0, input_num=0):
        #in_planes = self.in_planes
        if input_num>0:
            self.in_planes = input_num
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  clinical_num=clinical_num,
                  idx = 0))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes,clinical_num=clinical_num, idx = i))

        return nn.Sequential(*layers)

    def forward(self, x,sa=None):
        if sa is not None:
            x = torch.cat((x,sa),dim=1)

        x = self.attn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


#def generate_model(model_depth, **kwargs):
def generate_model(model_depth, **kwargs):#for filter
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model





