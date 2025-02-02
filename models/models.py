import math

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import copy
from .senet import *
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MHA(nn.Module):
    def __init__(self, n_dims, heads=4):
        super(MHA, self).__init__()
        self.query = nn.Linear(n_dims, n_dims)
        self.key = nn.Linear(n_dims, n_dims)
        self.value = nn.Linear(n_dims, n_dims)

        self.mha = torch.nn.MultiheadAttention(n_dims, heads)
        print('debug')

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out = self.mha(q,k,v)

        return out


## Transformer Block
##multi Head attetnion from BoTnet https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
class MHSA(nn.Module):
    def __init__(self, n_dims, width=16, height=16, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        ### , bias = False in conv2d
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size() # C // self.heads,
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, torch.div(C, self.heads, rounding_mode='floor'), -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

###also from https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py

class Bottleneck_Transformer(nn.Module):

    def __init__(self, in_planes, planes, stride=1, heads=4, resolution=None, use_mlp = False, expansion = 4):
        super(Bottleneck_Transformer, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList()
        self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
        if stride == 2:
            self.conv2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.use_MLP = use_mlp
        if use_mlp:
            self.LayerNorm = torch.nn.InstanceNorm2d(in_planes)
            self.MLP_torch = torchvision.ops.MLP(in_planes, [512, 2048])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.use_MLP:
            residual = out
            out = self.LayerNorm(out)
            out = out.permute(0,3,2,1)
            out = self.MLP_torch(out)
            out = out.permute(0,3,2,1)
            out = out + residual
            # out = F.relu(out)
        return out





# Defines the new fc layer and classification layer
# |--MLP--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.0, relu=False, bnorm=True, linear=False, return_f = True, circle=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.circle = circle
        add_block = []
        if linear: ####MLP to reduce
            final_dim = linear
            add_block += [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, final_dim)]
        else:
            final_dim = input_dim
        if bnorm:
            tmp_block = nn.BatchNorm1d(final_dim)
            tmp_block.bias.requires_grad_(False) 
            add_block += [tmp_block]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(final_dim, class_num, bias=False)] # 
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        if x.dim()==4:
            x = x.squeeze().squeeze()
        if x.dim()==1:
            x = x.unsqueeze(0)
        x = self.add_block(x)
        if self.return_f:
            f = x
            if self.circle:
                x = F.normalize(x)
                self.classifier[0].weight.data = F.normalize(self.classifier[0].weight, dim=1)
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x



class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class SEBottleneck(nn.Module):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, reduction=1, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SEModule(planes, reduction=reduction)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + self.shortcut(residual)
        out = self.relu(out)

        return out




class SEResNeXtBottleneck(nn.Module):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups=32, reduction=16, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        out = self.se_module(out) + self.shortcut(residual)
        out = self.relu(out)

        return out
class FFM(nn.Module):
    def __init__(self, inplanes=2048):
        super(FFM, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.Sqeeze_L = nn.Linear(in_features=2*inplanes, out_features=inplanes//2)
        self.Excitation_L = nn.Linear(in_features=inplanes // 2, out_features=inplanes)
        self.sigmiod = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.Sqeeze_G = nn.Linear(in_features=2*inplanes, out_features=inplanes//2)
        self.Excitation_G = nn.Linear(in_features=inplanes//2, out_features=inplanes)
    def forward(self, L, G):
        x_l = self.GAP(L)
        x_g = self.GAP(G)
        X = torch.cat((x_l, x_g), dim=1)
        X = X.squeeze()
        x_l = self.Sqeeze_L(X)
        x_l = self.relu(x_l)
        x_l = self.Excitation_L(x_l)
        x_l = self.sigmiod(x_l)
        x_l = x_l.view(x_g.shape[0], x_g.shape[1], 1, 1)

        x_g = self.Sqeeze_G(X)
        x_g = self.relu(x_g)
        x_g = self.Excitation_G(x_g)
        x_g = self.sigmiod(x_g)
        x_g = x_g.view(x_g.shape[0], x_g.shape[1], 1, 1)

        return L*x_l + L, G*x_g + G
class MHSA_AT(nn.Module):
    def __init__(self, n_dims, width=16, height=16, heads=4):
        super(MHSA_AT, self).__init__()
        self.heads = heads
        ### , bias = False in conv2d
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size() # C // self.heads,
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, torch.div(C, self.heads, rounding_mode='floor'), -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out, attention
class BOT_AT(nn.Module):

    def __init__(self, in_planes, planes, stride=1, heads=4, resolution=None, use_mlp = False, expansion = 4):
        super(BOT_AT, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList()
        self.conv2.append(MHSA_AT(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
        if stride == 2:
            self.conv2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.use_MLP = use_mlp
        if use_mlp:
            self.LayerNorm = torch.nn.InstanceNorm2d(in_planes)
            self.MLP_torch = torchvision.ops.MLP(in_planes, [512, 2048])

    def before_atten(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, attention = self.conv2(out)
        out= F.relu(self.bn2(out))
        return x, out, attention

    def after_atten(self, out, residual, attenion_l):
        out = self.bn3(self.conv3(out))
        out = out +  self.shortcut(residual) + out * attenion_l
        out = F.relu(out)
        if self.use_MLP:
            residual = out
            out = self.LayerNorm(out)
            out = out.permute(0, 3, 2, 1)
            out = self.MLP_torch(out)
            out = out.permute(0, 3, 2, 1)
            out = out + residual
            # out = F.relu(out)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, attention = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.use_MLP:
            residual = out
            out = self.LayerNorm(out)
            out = out.permute(0,3,2,1)
            out = self.MLP_torch(out)
            out = out.permute(0,3,2,1)
            out = out + residual
            # out = F.relu(out)
        return out

class SEModule_AT(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule_AT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        attention = self.sigmoid(x)
        return module_input * attention, attention

class SE_AT(nn.Module):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups=32, reduction=16, stride=1,
                 downsample=None, base_width=4):
        super(SE_AT, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule_AT(planes * 4, reduction=reduction)
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, attention):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = out + (torch.matmul(out.reshape(out.shape[0], 4, 256, 256), attention.permute(0, 1, 3, 2))).reshape(
            out.shape[0], 1024, 16, 16)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out, attention = self.se_module(out)
        out = out + self.shortcut(residual)
        out = self.relu(out)

        return out, attention


class SEBoT_Bottleneck(nn.Module):
    def __init__(self, in_features=1024, bottle_features=512):
        super(SEBoT_Bottleneck, self).__init__()
        self.transformer = BOT_AT(in_features, bottle_features, resolution=[16, 16], use_mlp=False)
        self.senet = SE_AT(in_features, bottle_features)

    def forward(self, L, G):
        resual_g ,out_g, attention_g = self.transformer.before_atten(G)
        out_l, attention_l = self.senet(L, attention_g)
        out_g = self.transformer.after_atten(out_g, resual_g, attention_l)

        return out_l, out_g

class SEBoT(nn.Module):


    def __init__(self, in_features=1024,bottle_features=512):
        super(SEBoT, self).__init__()
        self.layer1 = SEBoT_Bottleneck(in_features, bottle_features)
        self.layer2 = SEBoT_Bottleneck(bottle_features*4, bottle_features)
        self.layer3 = SEBoT_Bottleneck(bottle_features*4, bottle_features)

    def forward(self,x):
        L, G = self.layer1(x, x)
        L, G = self.layer2(L, G)
        L, G = self.layer3(L, G)
        return L, G

class base_branches(nn.Module):
    def __init__(self, backbone="ibn", stride=1):
        super(base_branches, self).__init__()
        if backbone == 'r50':
            model_ft = models.resnet50()
        elif backbone == '101ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        elif backbone == '34ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        else:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
            
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            if backbone == "34ibn":
                model_ft.layer4[0].conv1.stride = (1,1)
            else:
                model_ft.layer4[0].conv2.stride = (1,1)

        self.model = torch.nn.Sequential(*(list(model_ft.children())[:-3])) 

    def forward(self, x):
        x = self.model(x)
        return x
    
class multi_branches(nn.Module):
    def __init__(self, n_branches):
        super(multi_branches, self).__init__()

        model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        model_ft= model_ft.layer4
        self.n_branches = n_branches
        self.model = nn.ModuleList()

        if len(n_branches) > 0:
            for item in n_branches:
                if item == "R50":
                    self.model.append(copy.deepcopy(model_ft))
                elif item == "BoT":
                    layer_0 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp=False)
                    layer_1 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp=False)
                    layer_2 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp=False)
                    self.model.append(nn.Sequential(layer_0, layer_1, layer_2))
                elif item == 'SE':
                    se_layer1 = SEResNeXtBottleneck(inplanes=1024, planes=512, stride=2)
                    se_layer2 = SEResNeXtBottleneck(inplanes=2048, planes=512)
                    se_layer3 = SEResNeXtBottleneck(inplanes=2048, planes=512)
                    se = nn.Sequential(se_layer1, se_layer2, se_layer3)
                    self.model.append(se)
                elif item == 'SE-BoT':
                    self.model.append(SEBoT())
                else:
                    print("No valid architecture selected for branching by expansion!")
        else:
            self.model.append(model_ft)


    def forward(self, x):
        output = []
        for cnt, branch in enumerate(self.model):
            if self.n_branches[0] == 'SE-BoT':
                out1, out2 = branch(x)
                output.append(out1)
                output.append(out2)
            else:
                output.append(branch(x))
        return output




class FinalLayer(nn.Module):
    def __init__(self, class_num, n_branches, losses="LBS", droprate=0, linear_num=False, return_f = True, circle_softmax=False, n_cams=0, n_views=0, LAI=False):
        super(FinalLayer, self).__init__()    
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.finalblocks = nn.ModuleList()
        self.withLAI = LAI
        self.losses = losses

        for i in range(len(losses)):
            if(losses[i] == "tri"):
                bn = nn.BatchNorm1d(int(2048))
                bn.bias.requires_grad_(False)
                bn.apply(weights_init_kaiming)
                self.finalblocks.append(bn)
            else:
                self.finalblocks.append(
                    ClassBlock(2048, class_num, droprate, linear=linear_num, return_f=return_f,
                               circle=circle_softmax))
        if self.withLAI:
            # self.LAI = []
            self.n_cams = n_cams
            self.n_views = n_views
            if n_cams>0 and n_views>0:
                self.LAI = nn.Parameter(torch.zeros(len(losses), n_cams * n_views, 2048))
            elif n_cams>0:
                self.LAI = nn.Parameter(torch.zeros(len(losses), n_cams, 2048))
            elif n_views>0:
                self.LAI = nn.Parameter(torch.zeros(len(losses), n_views, 2048))
            else:
                self.withLAI = False



    def forward(self, x, cam, view):
        # if len(x) != len(self.finalblocks):
        #     print("Something is wrong")
        embs = []
        ffs = []
        preds = []
        for i in range(len(x)):
            emb = self.avg_pool(x[i]).squeeze(dim=-1).squeeze(dim=-1)
            if self.withLAI:
                if self.n_cams > 0 and self.n_views >0:
                    lai = self.LAI[i, cam * self.n_views + view, :].squeeze()
                    emb = emb + lai
                elif self.n_cams >0:
                    emb = emb + self.LAI[i, cam, :]
                else:
                    emb = emb + self.LAI[i, view, :]

            if self.losses[i] == 'tri':
                ff = self.finalblocks[i](emb)
                embs.append(emb)
                ffs.append(ff)
            elif self.losses[i] == 'ce':
                pred, ff = self.finalblocks[i](emb)
                ffs.append(ff)
                preds.append(pred)
            elif self.losses[i] == 'ce+tri':
                pred, ff = self.finalblocks[i](emb)
                embs.append(emb)
                ffs.append(ff)
                preds.append(pred)
            else:
                print("No valid loss selected!!!")
                    
        return preds, embs, ffs

    
class KLMIN_model(nn.Module):
    def __init__(self, class_num, n_branches, losses="LBS", backbone="ce", droprate=0, linear_num=False, return_f = True, circle_softmax=False, LAI=False, n_cams=0, n_views=0):
        super(KLMIN_model, self).__init__()
        self.modelup2L3 = base_branches(backbone=backbone)
        self.modelL4 = multi_branches(n_branches=n_branches)
        self.finalblock = FinalLayer(class_num=class_num, n_branches=n_branches, losses=losses, droprate=droprate, linear_num=linear_num, return_f=return_f, circle_softmax=circle_softmax, LAI=LAI, n_cams=n_cams, n_views=n_views)
        

    def forward(self, x,cam, view):
        mix = self.modelup2L3(x)
        output = self.modelL4(mix)
        preds, embs, ffs = self.finalblock(output, cam, view)

        return preds, embs, ffs, output



if __name__ == "__main__":
    model = KLMIN_model(575, n_branches=["se-bot", "se-bot"], losses=["ce", "ce", "tri", "tri"], LAI=True, n_cams=20,
                        n_views=8)
    print(model)
    print(count_parameters(model))



