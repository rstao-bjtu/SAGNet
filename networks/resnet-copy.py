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
        # self.conv1_2 = nn.Conv2d(self.weight1.shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_ = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_ = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # print('+'*15)
        # print(self.conv1.weight.requires_grad)
        # print(self.bn1.weight.requires_grad)
        # print(self.bn1.bias.requires_grad)
        # print(self.bn1.running_mean.requires_grad)
        # print(self.bn1.running_var.requires_grad)
        # print('+'*15)

        # self.conv1.weight.requires_grad        = False
        # self.bn1.weight.requires_grad       = False
        # self.bn1.bias.requires_grad         = False
        # self.bn1.running_mean.requires_grad = False
        # self.bn1.running_var.requires_grad  = False     

        # print('+'*15)
        # print(self.conv1.weight.requires_grad)
        # print(self.bn1.weight.requires_grad)
        # print(self.bn1.bias.requires_grad)
        # print(self.bn1.running_mean.requires_grad)
        # print(self.bn1.running_var.requires_grad)
        # print('+'*15)
        
        self.inplanes = 64
        

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            # 256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 512
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        
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
        x_group = torch.split(x, split_size, dim=1)
        [print(i.shape) for i in x_group];print()
        # print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x_group_freq = [self.hfreqC(x, 4) for x in x_group]
        
        #torch.cuda.empty_cache()
        [print(i.shape, i.device) for i in x_group_freq]

        return torch.cat(list(x_group_freq ,1)


    def hfreqWH_group(self, x, group):
        b,c,h,w = x.shape
        assert h > group and w > group

        subgroup_w = w//group
        subgroup_h = h//group

        split_size_w = [subgroup_w]*group; split_size_w[-1] = subgroup_w + w%group
        split_size_h = [subgroup_h]*group; split_size_h[-1] = subgroup_h + h%group

        assert sum(split_size_w) == w and sum(split_size_h) == h 

        w_group = torch.split(x, split_size_w, dim=3)
        wh_group= [torch.split(wgroup, split_size_w, dim=2) for wgroup in w_group]

        x_group_freq = []
        for h_groups in wh_group:
            x_group_freq.append( torch.cat([ self.hfreqWH(h_group, 4) for h_group in h_groups ], 2) )

        return torch.cat(x_group_freq ,3)


    def forward(self, x):
        
        # x = F.conv2d(x, self.weight1, stride=1, padding=1)
        # x = F.batch_norm(x, self.bn_running_mean, self.bn_running_var, self.bn_weight, self.bn_bias, training=False, eps=0.001)
        # x = F.relu(x, inplace=True)
        x = self.conv1_(x) #64
        x = self.bn1_(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.hfreqC_group(x, 4)
        x2 = self.hfreqWH_group(x, 4)
        x = torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2) + 1e-6)#.clone().detach()
        x = self.layer1(x)# in64 out256
        # x = self.hfreqC(x)
        x = self.layer2(x)# in256  out512
        # x = self.hfreqC(x)
        x = self.layer3(x)# in512  out1024
        x = self.layer4(x)# in1024 out2048

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

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
    # if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
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
