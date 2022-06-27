import torch.nn as nn
import torch
from spectral_normalization import SpectralNorm
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F
import functools



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv2, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

#空间注意力
class ChannelPool(nn.Module):
    def forward(self, x):#计算最大池化和平均池化，并连接
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self,norm):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv2(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        if norm == "batch":
            self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CoupledWeight(nn.Module):
    def forward(self, x1,x2):

        x1 = x1.unsqueeze(2).unsqueeze(3)
        x2 = x2.unsqueeze(2).unsqueeze(3)


        concat = torch.cat([x1,x2],3)

        coupled_weights = F.softmax(concat,dim=3)
        ir_weights = coupled_weights[:,:,:,0].unsqueeze(dim=3)
        vi_weights = coupled_weights[:, :, :, 1].unsqueeze(dim=3)

        #print(ir_weights[:,30,:,:])
        #print(vi_weights[:, 30, :, :])

        return ir_weights,vi_weights




class FusionBlock(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8,norm = "instance"):
        super(FusionBlock, self).__init__()
        self.SpatialGate = SpatialGate(norm)
        self.ChannelGate = ChannelGate(gate_channels)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )



    def forward(self, ir,vi):
        #获得分别的空间注意力张量,注意到空间上的显著特征

        #ir = self.ChannelGate(ir)
        #vi = self.ChannelGate(ir)


        ir_s = self.SpatialGate(ir)
        vi_s = self.SpatialGate(vi)



        #为红外和可见光张量计算对应通道相加的融合权重
        ir_pool = F.avg_pool2d(ir_s, (ir_s.size(2), ir_s.size(3)), stride=(ir_s.size(2), ir_s.size(3)))
        vi_pool = F.avg_pool2d(vi_s, (vi_s.size(2), vi_s.size(3)), stride=(vi_s.size(2), vi_s.size(3)))

        ir_v = self.mlp(ir_pool)
        vi_v = self.mlp(vi_pool)

        ir_w,vi_w = CoupledWeight()(ir_v,vi_v)

        #融合特征
        x_out= ir_w * ir_s + vi_w * vi_s

        return x_out



class MultiContext_bridge4(nn.Module):
    def __init__(self,nl_layer,norm_layer, in_dims, mid_dims, rate=[3, 5, 7]):
        super().__init__()

        pad = [nn.ReflectionPad2d(1)]

        down_conv = [nl_layer()] + pad + [SpectralNorm(nn.Conv2d(in_dims, mid_dims,
                                                           kernel_size=4, stride=2, padding=0))] + [norm_layer(mid_dims)] #16*16 512

        self.conv =  nn.Sequential(*down_conv)

        self.block1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                mid_dims, mid_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            )),
            nl_layer(),
            norm_layer(mid_dims),
        )
        self.block2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                mid_dims, mid_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            )),
            nl_layer(),
            norm_layer(mid_dims),
        )
        self.block3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                mid_dims, mid_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            )),
            nl_layer(),
            norm_layer(mid_dims),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.output = SpectralNorm(nn.Conv2d((len(rate)+1) * mid_dims,mid_dims, 1))


    def forward(self, x):
        #获取上下文/语义高级信息，由于高度抽象，因此不需要考虑空间注意力，只需要考虑通道上的语义信息的重要性
        #
        x = self.conv(x)

        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.avg_pool(x)

        #x4 = torch.ones((x1.shape[0],x.shape[IR],x1.shape[VI],x2.shape[3])).cuda()*x4
        x4 = self.avg_pool(x).expand_as(x1)

        out = self.output(torch.cat([x1, x2, x3, x4], dim=1)) + x


        #print(x.size())

        return out




class UpFusionBlock(nn.Module):
    def __init__(self, high_dim,low_dim,up_factor = 1 ,upsample='nearest', reduction_ratio=8):
        super(UpFusionBlock, self).__init__()

        self.up = False
        #统一纬度
        self.sequeeze_conv = nn.Conv2d(high_dim,low_dim,1,1,0)
        #上采样统一大小
        if up_factor != 1:
            self.up = True
            if upsample == 'bilinear':
                self.upsample = nn.Upsample(scale_factor=up_factor, mode='bilinear')
            elif upsample == 'nearest':
                self.upsample = nn.Upsample(scale_factor=up_factor, mode='nearest')

        #池化

        #瓶颈压缩计算量,同时初步提取高级语义特征的依赖以及当前低级特征可能存在的依赖
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(low_dim, low_dim // reduction_ratio),
            nn.ReLU(),
        )
        #element-wise add

        #在高级语义特征的辅助下，计算出调整过的低级特征各个通道之间的语义依赖性
        self.inter_dependency = nn.Linear(low_dim // reduction_ratio, low_dim)



    def forward(self, l_f,h_f):

        if self.up == True:
            h_f = self.upsample(h_f)
        h_f = self.sequeeze_conv(h_f)

        l_pool = F.avg_pool2d(l_f, (l_f.size(2), l_f.size(3)), stride=(l_f.size(2), l_f.size(3)))
        h_pool = F.avg_pool2d(h_f, (h_f.size(2), h_f.size(3)), stride=(h_f.size(2), h_f.size(3)))

        l_v = self.mlp(l_pool)
        h_v = self.mlp(h_pool)

        d_w = torch.sigmoid(self.inter_dependency(l_v+h_v)).unsqueeze(2).unsqueeze(3)



        return d_w*l_f




class UpFusionBlock5(nn.Module):
    def __init__(self, high_dim,low_dim,up_factor = 1 ,upsample='nearest', reduction_ratio=16):
        super(UpFusionBlock5, self).__init__()

        self.up = False
        #统一纬度
        #self.sequeeze_conv = nn.Conv2d(high_dim,low_dim,IR,IR,0)
        #上采样统一大小
        if up_factor != 1:
            self.up = True
            if upsample == 'bilinear':
                self.upsample = nn.Upsample(scale_factor=up_factor, mode='bilinear')
            elif upsample == 'nearest':
                self.upsample = nn.Upsample(scale_factor=up_factor, mode='nearest')

        #池化

        self.sequeeze_conv = nn.Conv2d(high_dim+low_dim,low_dim,1,1,0)

        #瓶颈压缩计算量,同时初步提取高级语义特征的依赖以及当前低级特征可能存在的依赖
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(low_dim, low_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(low_dim // reduction_ratio, low_dim)
        )
        #element-wise add



    def forward(self, l_f,h_f):

        if self.up == True:
            h_f = self.upsample(h_f)

        hl = torch.cat([h_f,l_f],1)

        hl = self.sequeeze_conv(hl)

        hl_mean_pool = F.avg_pool2d(hl, (hl.size(2), hl.size(3)), stride=(hl.size(2), hl.size(3)))
        hl_max_pool = F.max_pool2d(hl, (hl.size(2), hl.size(3)), stride=(hl.size(2), hl.size(3)))

        hl_mean = self.mlp(hl_mean_pool)
        hl_max = self.mlp(hl_max_pool)
        
        hl = hl_mean + hl_max


        d_w = torch.sigmoid(hl.unsqueeze(2).unsqueeze(3))



        return d_w*l_f