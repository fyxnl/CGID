import torch.nn as nn
import math
import torch
import Res2Net as Pre_Res2Net
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class XIAOBO(nn.Module):
    def __init__(self):
        super(XIAOBO, self).__init__()

        self.mns = MainNetworkStructure(3,4)

    def forward(self, x):
        Fout,z= self.mns(x)

        return Fout,z


class MainNetworkStructure(nn.Module):
    def __init__(self, inchannel, channel):
        super(MainNetworkStructure, self).__init__()
        self.cfceb_l = CCEM(inchannel,channel)
        self.cfceb_m = CCEM(2*channel,channel * 4)
        self.cfceb_s = CCEM(channel * 8,channel * 16)
        self.ein = BRB(channel)
        self.el = BRB(channel)
        self.em = BRB(channel * 4)
        self.es = BRB(channel * 16)
        self.ds = BRB(channel * 16)
        self.dm = BRB(channel * 4)
        self.dl = BRB(channel)

        self.conv_eltem = nn.Conv2d(channel,4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_emtes = nn.Conv2d(4 * channel, 16 * channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_r_eltem = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_r_emtes = nn.Conv2d(4 * channel, 8 * channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_g_eltem = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_g_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)

        self.conv_b_eltem = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_d_emtem= nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)

        self.conv_dstdm = nn.Conv2d(16 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_dmtdl = nn.Conv2d(4 * channel, channel, kernel_size=1, stride=1, padding=0, bias=False)

        # self.conv_r_in = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_g_in = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_b_in = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_in = nn.Conv2d(inchannel, channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_out = nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def forward(self, x):
        c=x
        # r = self.conv_r_in(x[:, 0, :, :].unsqueeze(1))
        # g = self.conv_g_in(x[:, 1, :, :].unsqueeze(1))
        # b = self.conv_b_in(x[:, 2, :, :].unsqueeze(1))
        x_ll, x_hl, x_lh, x_hh, x_out_1 = self.cfceb_l(x)
        cc=self.conv_r_eltem(self.maxpool(x_ll))
        x_r_m,_,_,_,_ = self.cfceb_m(self.conv_r_eltem(self.maxpool(x_ll)))
        _,x_g_m,_,_,_=self.cfceb_m(self.conv_g_eltem(self.maxpool(x_hl)))
        _,_,x_b_m,_,_= self.cfceb_m(self.conv_b_eltem(self.maxpool(x_lh)))
        _,_,_,x_d_m,x_out_m=self.cfceb_m(self.conv_d_emtem(self.maxpool(x_hh)))
        # x_r_m, x_g_m, x_b_m, x_d_m,x_out_m = self.cfceb_m(self.conv_r_eltem(self.maxpool(x_ll)),
        #                                             self.conv_g_eltem(self.maxpool(x_hl)),
        #                                             self.conv_b_eltem(self.maxpool(x_lh)),
        #                                             self.conv_d_emtem(self.maxpool(x_hh))                                                    )
        _, _, _,_, x_out_s = self.cfceb_s(self.conv_r_emtes(self.maxpool(x_r_m)))

        x_elin = self.ein(self.conv_in(x))
        cc=x_elin * x_out_1
        elout = self.el(x_elin * x_out_1)
        x_emin = self.conv_eltem(self.maxpool(elout))
        emout = self.em(x_emin * x_out_m)
        x_esin = self.conv_emtes(self.maxpool(emout))
        esout = self.es(x_esin * x_out_s)
        dsout = self.ds(esout)
        x_dmin = self._upsample(self.conv_dstdm(dsout), emout) + emout
        dmout = self.dm(x_dmin)
        x_dlin = self._upsample(self.conv_dmtdl(dmout), elout) + elout
        dlout = self.dl(x_dlin)
        x_out = self.conv_out(dlout)+x

        return x_out,dsout




def dwt_init1(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL,x_HL,x_LH,x_HH


class DWT1(nn.Module):
    def __init__(self):
        super(DWT1, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init1(x)


class CCEM(nn.Module):
    def __init__(self,inchannel,channel):
        super().__init__()
        self.dwt1 = DWT1()
        self.bb_ll = BRB(channel)
        self.bb_hl = BRB(channel)
        self.bb_lh = BRB(channel)
        self.bb_hh = BRB(channel)

        self.cab = CAB(2 * channel)
        self.cab_RGB = CAB(4 * channel)
        self.conv1x1_ll = nn.Conv2d(inchannel,channel, kernel_size=1, padding=0)
        self.conv1x1_hl = nn.Conv2d(inchannel, channel, kernel_size=1, padding=0)
        self.conv1x1_lh = nn.Conv2d(inchannel,channel, kernel_size=1, padding=0)
        self.conv1x1_hh = nn.Conv2d(inchannel, channel, kernel_size=1, padding=0)
        self.conv_out1 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out2 = nn.Conv2d(channel * 6, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out3 = nn.Conv2d(channel * 4, 6*channel, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        dwt_ll,dwt_hl, dwt_lh,dwt_hh = self.dwt1(x)
        dwt_ll_frequency = self.conv1x1_ll(dwt_ll)
        dwt_hl_frequency = self.conv1x1_hl(dwt_hl)
        dwt_lh_frequency = self.conv1x1_lh(dwt_lh)
        dwt_hh_frequency = self.conv1x1_hh(dwt_hh)
        x_ll=self.bb_ll(dwt_ll_frequency)
        x_hl=self.bb_hl(dwt_hl_frequency)
        x_lh =self.bb_lh(dwt_lh_frequency)
        x_hh=self.bb_hh(dwt_hh_frequency )
        x_lowhigh_a=self.conv_out1(self.cab(torch.cat((x_ll,x_hl), 1)))
        x_lowhigh_b = self.conv_out1(self.cab(torch.cat((x_ll, x_lh), 1)))
        x_lowhigh_c = self.conv_out1(self.cab(torch.cat((x_ll, x_hh), 1)))
        x_highhigh_a = self.conv_out1(self.cab(torch.cat((x_hl, x_lh), 1)))
        x_lighhigh_b = self.conv_out1(self.cab(torch.cat((x_hl, x_hh), 1)))
        x_lighhigh_c = self.conv_out1(self.cab(torch.cat((x_lh, x_hh), 1)))
        x_idwt= self.cab_RGB(torch.cat((x_ll,x_hl,x_lh,x_hh),1))
        x_idwt=self.conv_out3(x_idwt)
        x_out = self.conv_out2(torch.cat((x_lowhigh_a, x_lowhigh_b, x_lowhigh_c,x_highhigh_a,x_lighhigh_b,x_lighhigh_c), 1) + x_idwt)
        return x_ll, x_hl, x_lh, x_hh, x_out


class BRB(nn.Module):
    def __init__(self, channel, norm=False):
        super(BRB, self).__init__()

        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)

        self.conv_out = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.act = nn.PReLU(channel)

        self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)  # nn.InstanceNorm2d(channel)#

    def forward(self, x):
        x_1 = self.act(self.norm(self.conv_1(x)))
        x_2 = self.act(self.norm(self.conv_2(x_1)))
        x_out = self.act(self.norm(self.conv_out(x_2)) + x)
        # x_out1=self.act(self.norm(self.conv_1(x_out)))
        # x_out2 = self.act(self.norm(self.conv_out(x_out1)+x))
        return x_out

class CAB(nn.Module):
    def __init__(self, k_size=3):
        super(CAB, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x) + self.max_pool(x)
        y_temp = self.conv(y.squeeze(-1).transpose(-1, -2))
        y_temp = y_temp.unsqueeze(-1)
        y = y.transpose(-2, -3)
        y_temp = F.upsample(y_temp, y.size()[2:], mode='bilinear')
        y = y_temp.transpose(-2, -3)

        camap = self.sigmoid(y)

        return camap
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_init = self.relu(x)
        x = self.maxpool(x_init)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_output= self.layer3(x_layer2)
        return x_init, x_layer1, x_layer2, x_output

class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class knowledge_adaptation_UNet(nn.Module):
    def __init__(self):
        super(knowledge_adaptation_UNet, self).__init__()
        self.encoder = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101 = Pre_Res2Net.Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101.load_state_dict(torch.load('./weights/res2net101_v1b_26w_4s-0812c246.pth'))
        pretrained_dict = res2net101.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)
        self.up_block= nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3)
        self.attention1 = CP_Attention_block(default_conv, 256, 3)
        self.attention2 = CP_Attention_block(default_conv, 192, 3)
        self.attention3 = CP_Attention_block(default_conv, 112, 3)
        self.attention4 = CP_Attention_block(default_conv, 44, 3)
        self.conv_process_1 = nn.Conv2d(44, 44, kernel_size=3,padding=1)
        self.conv_process_2 = nn.Conv2d(44, 28, kernel_size=3,padding=1)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(28, 3, kernel_size=7, padding=0), nn.Tanh())
    def forward(self, input):
        x_inital, x_layer1, x_layer2, x_output = self.encoder(input)
        x_mid = self.attention0(x_output)
        x = self.up_block(x_mid)
        x = self.attention1(x)
        x = torch.cat((x, x_layer2), 1)
        x = self.up_block(x)
        x = self.attention2(x)
        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = self.attention3(x)
        x = torch.cat((x, x_inital), 1)
        x = self.up_block(x)
        x = self.attention4(x)
        x=self.conv_process_1(x)
        out=self.conv_process_2(x)
        return out,x_output

class DWT_transform(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels*3, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        dwt_low_frequency,dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency,dwt_high_frequency





class fusion_net(nn.Module):
    def __init__(self):
        super(fusion_net, self).__init__()
        # self.dwt_branch=dwt_UNet1()
        self.XIAOBO=XIAOBO()
        self.knowledge_adaptation_branch=knowledge_adaptation_UNet()
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(31, 3, kernel_size=7, padding=0), nn.Tanh())
    def forward(self, input):
        # dwt_branch,z=self.dwt_branch(input)
        dwt_branch, z = self.XIAOBO(input)
        knowledge_adaptation_branch,z1=self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        return x,z,z1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    net = fusion_net().to('cuda')
    summary(net, input_size=(3, 224, 224), batch_size=1)