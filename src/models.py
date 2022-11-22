import numpy as np
from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import e2cnn

from src.icosahedron import Icosahedron


class EqBasicBlock(nn.Module):
    def __init__(self, gspace, inplanes, planes, stride, downsample, norm_layer):
        super().__init__()
        in_type = e2cnn.nn.FieldType(gspace, inplanes*[gspace.regular_repr])
        out_type = e2cnn.nn.FieldType(gspace, planes*[gspace.regular_repr])

        self.conv1 = e2cnn.nn.R2Conv(in_type, out_type, 3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_layer(out_type)
        self.relu = e2cnn.nn.ReLU(out_type, True)
        self.conv2 = e2cnn.nn.R2Conv(out_type, out_type, 3, padding=1, stride=1, bias=False)
        self.bn2 = norm_layer(out_type)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EqEncoder(nn.Module):
    def __init__(self,
                 input_shape: tuple,
                 out_fdim: int,
                 order=4,
                 layers=[2,2,2,2],
                 strides=[2,2,1,1],
                 conv1_ksize=5,
                 pool_final=False,
                 perform_maxpool=True,
                 use_norm=True):
        self.out_fdim = out_fdim
        super().__init__()

        self.r2_act = e2cnn.gspaces.Rot2dOnR2(N=order)
        self.in_type = e2cnn.nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr])

        self.mask = e2cnn.nn.MaskModule(self.in_type, S=input_shape[-1])

        self.norm_layer = e2cnn.nn.InnerBatchNorm if use_norm else e2cnn.nn.IdentityModule
        self.dilation = 1
        self.base_width = 38
        self.inplanes = self.base_width

        self.input_shape = input_shape
        out_type = e2cnn.nn.FieldType(self.r2_act, self.base_width*[self.r2_act.regular_repr])
        self.conv1 = e2cnn.nn.R2Conv(self.in_type, out_type, conv1_ksize, stride=2,
                               padding=conv1_ksize//2, bias=not use_norm)
        self.bn1 = self.norm_layer(out_type)
        if perform_maxpool:
            self.maxpool = e2cnn.nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = e2cnn.nn.IdentityModule(out_type)

        self.relu1 = e2cnn.nn.ReLU(out_type, True)
        self.layer1 = self._make_layer(self.r2_act, self.base_width, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(self.r2_act, self.base_width*2, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(self.r2_act, self.base_width*4, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(self.r2_act, self.base_width*8, layers[3], stride=strides[3])

        if pool_final:
            self.final_pool = e2cnn.nn.PointwiseAdaptiveAvgPool(
                                    e2cnn.nn.FieldType(self.r2_act, self.base_width*8*[self.r2_act.regular_repr]),
                                    (1,1))
        else:
            self.final_pool = None

        self.gpool = e2cnn.nn.GroupPooling(e2cnn.nn.FieldType(self.r2_act, self.base_width*8*[self.r2_act.regular_repr]))
        self.out = e2cnn.nn.R2Conv(e2cnn.nn.FieldType(self.r2_act, self.base_width*8*[self.r2_act.trivial_repr]),
                                   e2cnn.nn.FieldType(self.r2_act, out_fdim*[self.r2_act.trivial_repr]),
                                   1)

    def _make_layer(self,
                    gspace: e2cnn.gspaces.GSpace,
                    planes: int,
                    blocks: int,
                    stride: int=1,
                    dilate: bool=False,
                   ) -> e2cnn.nn.SequentialModule:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            in_type = e2cnn.nn.FieldType(gspace, self.inplanes*[gspace.regular_repr])
            out_type = e2cnn.nn.FieldType(gspace, planes*[gspace.regular_repr])
            downsample = e2cnn.nn.SequentialModule(
                e2cnn.nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, stride=stride),
                self.norm_layer(out_type),
            )

        layers = []
        layers.append(
            EqBasicBlock(
                gspace, self.inplanes, planes, stride, downsample, self.norm_layer
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                EqBasicBlock(gspace, self.inplanes, planes, 1, None, self.norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = e2cnn.nn.GeometricTensor(x, self.in_type)
        x = self.mask(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.final_pool is not None:
            x = self.final_pool(x)

        x = self.gpool(x)
        x = self.out(x)

        return x.tensor

    @property
    def output_shape(self):
        x = torch.zeros((1,*self.input_shape), dtype=torch.float32)
        return self.forward(x).shape[1:]


class Encoder(nn.Module):
    # based on https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    def __init__(self,
                 input_shape: tuple,
                 out_fdim: int,
                 layers=[2,2,2,2],
                 strides=[2,2,1,1],
                 conv1_ksize=5,
                 perform_maxpool=True,
                 use_norm=True,
                 pool_final=False,
                 **kwargs,
                ):
        super().__init__()
        self.norm_layer = nn.BatchNorm2d if use_norm else nn.Identity
        self.dilation = 1
        self.inplanes = 64
        self.base_width = 64

        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(self.input_shape[0], self.inplanes, conv1_ksize, 2,
                               conv1_ksize//2, bias=not use_norm)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if perform_maxpool else nn.Identity()
        self.layer1 = self._make_layer( 64, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3])

        if pool_final:
            self.final_pool = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.final_pool = None

        self.out = nn.Conv2d(512, out_fdim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int=1, dilate: bool=False,
                   ) -> nn.Sequential:
        block = resnet.BasicBlock

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, 1,
                self.base_width, previous_dilation, self.norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=self.norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.final_pool is not None:
            x = self.final_pool(x)

        return self.out(x)

    @property
    def output_shape(self):
        x = torch.zeros((1,*self.input_shape), dtype=torch.float32)
        return self.forward(x).shape[1:]


class FmapSampler(nn.Module):
    '''Sample features from feature map onto half mesh
    '''
    def __init__(self,
                 fmap_size: int,
                 sigma: float=0.5,
                 fraction_coverage: float=0.9,
                 ico_level: int=0,
                ):
        super().__init__()
        ico = Icosahedron(level=ico_level)
        px, py = 0.5 * fmap_size * fraction_coverage * ico.halfmesh_locs.T + fmap_size/2
        self.px, self.py = px, py
        gridx, gridy = torch.meshgrid(2*[torch.arange(fmap_size)+0.5], indexing='ij')
        scale = 1 / np.sqrt(2 * np.pi * sigma**2)
        data = scale * torch.exp(-((gridx.unsqueeze(-1) - px).pow(2) + (gridy.unsqueeze(-1) - py).pow(2)) / (2*sigma**2) )
        # normalize
        data = data / data.sum((0,1), keepdims=True)
        data = data.unsqueeze(0).unsqueeze(0).to(torch.float32)

        self.weight = nn.Parameter(data=data, requires_grad=False)

    def forward(self, x):
        x = (x.unsqueeze(-1) * self.weight).sum((2,3))
        x = torch.transpose(x, 1, 2)
        return x
