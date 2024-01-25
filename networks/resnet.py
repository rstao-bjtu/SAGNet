import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
# from kornia.filters import Canny

# CUDA_VISIBLE_DEVICES=0 python train4.py --name 4class-resnet_car_cat_chair_horse_bs32_ --dataroot /root/ --classes car,cat,chair,horse --batch_size 32 --delr_freq 10 --lr 0.002 --niter 100 --delr 0.8 --pth random_hfreq

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out






class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, lnum=1):
        super(ResNet, self).__init__()
        
        self.printOne = 1
        self.lnum = lnum*2-1


        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)).cuda())
        self.bias1   = nn.Parameter(torch.randn((64,)).cuda())
        self.realconv1 = conv1x1(64, 64, stride=1)
        self.imagconv1 = conv1x1(64, 64, stride=1)

        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)).cuda())
        self.bias2   = nn.Parameter(torch.randn((64,)).cuda())
        self.realconv2 = conv1x1(64, 64, stride=1)
        self.imagconv2 = conv1x1(64, 64, stride=1)


        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias3   = nn.Parameter(torch.randn((256,)).cuda())
        self.realconv3 = conv1x1(256, 256, stride=1)
        self.imagconv3 = conv1x1(256, 256, stride=1)

        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias4   = nn.Parameter(torch.randn((256,)).cuda())
        self.realconv4 = conv1x1(256, 256, stride=1)
        self.imagconv4 = conv1x1(256, 256, stride=1)

        
        self.inplanes = 64
        

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            # 256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 512
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 1024
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        
        # x = F.conv2d(x, self.weight1, stride=1, padding=1)
        # x = F.batch_norm(x, self.bn_running_mean, self.bn_running_var, self.bn_weight, self.bn_bias, training=False, eps=0.001)
        # x = F.relu(x, inplace=True)
        x = self.hfreqWH(x, 4)

        x = F.conv2d(x, self.weight1, self.bias1, stride=1, padding=0)
        x = F.relu(x, inplace=True)

        x = self.hfreqC(x, 4)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight2, self.bias2, stride=2, padding=0)
        x = F.relu(x, inplace=True)
   
        x = self.hfreqC(x, 4)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)


        x = self.maxpool(x)
        x = self.layer1(x)# in64 out256

        x = F.conv2d(x, self.weight3, self.bias3, stride=1, padding=0)
        x = F.relu(x, inplace=True)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight4, self.bias4, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.layer2(x)# in256  out512


        # x = self.layer3(x)# in512  out1024
        # x = self.layer4(x)# in1024 out2048

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def hfreqWH(self, x, scale):
        assert scale>2
        #print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        b,c,h,w = x.shape
        # scale = 4
        x[:,:,h//2-h//scale:h//2+h//scale,w//2-w//scale:w//2+w//scale ] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        #print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        #print()
        return x

    def hfreqC(self, x, scale):
        assert scale>2
        # print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft(x, dim=1, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=1) 
        b,c,h,w = x.shape
        # scale = 4
        x[:,c//2-c//scale:c//2+c//scale,:,:] = 0.0
        # x[:,:c//scale,:,:] = 0.0
        # x[:,c-c//scale:,:,:] = 0.0

        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
 
        # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        # print()
        return x


    def hfreqC_group(self, x, group):
        b,c,h,w = x.shape
        assert c > group

        subgroup = c//group
        split_size = [subgroup]*group
        split_size[-1] = subgroup + c%group

        assert sum(split_size) == c
        split_size = list(range(0, c, subgroup))
        split_size[-1] = c

        scale = 4
        x_group_freq = []
        for split1,split2 in zip(split_size[:-1], split_size[1:]):
            xg = torch.fft.fft(x[:, split1:split2, :, :], dim=1, norm="ortho")#,norm='forward' "ortho"
            xg = torch.fft.fftshift(xg, dim=1) 
            b,c,h,w = xg.shape
            xg[:,c//2-c//scale:c//2+c//scale,:,:] = 0.0
            xg = torch.fft.ifftshift(xg, dim=1)
            xg = torch.fft.ifft(xg, dim=1, norm="ortho")
            xg = torch.real(xg)
            xg = F.relu(xg, inplace=True)
            # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
            x_group_freq.append(xg)
        
        # print('hfreqC_group')
        # [print(i.shape, i.device) for i in x_group_freq]
        x = torch.cat(x_group_freq ,1)
        return x 


    def hfreqWH_group(self, x, group):
        b,c,h,w = x.shape
        assert h > group and w > group

        subgroup_w = w//group
        subgroup_h = h//group

        split_size_w = [subgroup_w]*group; split_size_w[-1] = subgroup_w + w%group
        split_size_h = [subgroup_h]*group; split_size_h[-1] = subgroup_h + h%group

        assert sum(split_size_w) == w and sum(split_size_h) == h 

        w_group = torch.split(x, split_size_w, dim=3)
        # [print(i.shape, i.device) for i in w_group];print()

        wh_group= [torch.split(wgroup, split_size_w, dim=2) for wgroup in w_group]

        x_group_freq = []
        for h_groups in wh_group:
            x_group_freq.append( torch.cat([ self.hfreqWH(h_group, 4) for h_group in h_groups ], 2) )

        # [print(i.shape, i.device) for i in x_group_freq]
        x = torch.cat(x_group_freq ,3)
        # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        # print(x.shape) 
        return x





def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
