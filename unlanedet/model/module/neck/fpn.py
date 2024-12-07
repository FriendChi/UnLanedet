import math
import warnings
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ....layers import Conv2d,get_norm,Activation
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')
    
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x*psi
class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention=False,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier',
                               layer='Conv2d',
                               distribution='uniform'),
                 cfg=None):
        super(FPN, self).__init__()
        # assert isinstance(in_channels, list)
        self.in_channels = in_channels  # 保存输入通道列表
        self.out_channels = out_channels  # 保存输出通道数
        self.num_ins = len(in_channels)  # 输入的特征图数量（等于 in_channels 的长度）
        self.num_outs = num_outs  # 输出特征图的层数

        self.backbone_end_level = self.num_ins  # 设置骨干网络的结束层（即输入的层数）
        self.start_level = 0  # 设置起始层，通常为 0
        self.lateral_convs = nn.ModuleList()  # 用于存储 lateral 卷积层的列表
        self.fpn_convs = nn.ModuleList()  # 用于存储 FPN 卷积层的列表
        self.galas = nn.ModuleList()
        # 初始化 lateral 卷积和 FPN 卷积层
        for i in range(self.start_level, self.backbone_end_level):
            #横向卷积层,1*1卷积用于保持通道统一
            l_conv = ConvModule(
                in_channels[i],  # 输入通道数
                out_channels,  # 输出通道数
                1,  # 卷积核大小为 1
                conv_cfg=None,  # 卷积配置（未指定）
                norm_cfg=None,  # 归一化配置（未指定）
                act_cfg=None,  # 激活函数配置（未指定）
                inplace=False,
            )
            # FPN 卷积层，处理后的通道和尺寸都不变
            fpn_conv = ConvModule(
                out_channels,  # 输入通道数为输出通道数
                out_channels,  # 输出通道数
                3,  # 卷积核大小为 3
                padding=1,  # padding 为 1
                conv_cfg=None,  # 卷积配置（未指定）
                norm_cfg=None,  # 归一化配置（未指定）
                act_cfg=None,  # 激活函数配置（未指定）
                inplace=False,
            )

            self.lateral_convs.append(l_conv)  # 将 lateral 卷积层添加到列表中
            self.fpn_convs.append(fpn_conv)  # 将 FPN 卷积层添加到列表中
            
        for i in range(len(self.lateral_convs)-1):
            self.galas.append(LGAG(out_channels, out_channels, out_channels))
    def forward(self, inputs):
        """
        Args:
            inputs (List[torch.Tensor]): Input feature maps.
              Example of shapes:
                ([1, 64, 80, 200], [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]).
        Returns:
            outputs (Tuple[torch.Tensor]): Output feature maps.
              The number of feature map levels and channels correspond to
               `num_outs` and `out_channels` respectively.
              Example of shapes:
                ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        """
        if type(inputs) == tuple:  # 如果输入是 tuple 类型，将其转换为 list 类型
            inputs = list(inputs)

        assert len(inputs) >= len(self.in_channels)  # 确保输入的特征图数量不小于 in_channels 的长度

        if len(inputs) > len(self.in_channels):  # 如果输入的特征图数量大于 in_channels 的长度
            for _ in range(len(inputs) - len(self.in_channels)):  # 删除多余的输入特征图
                del inputs[0]

        # 构建 lateral 卷积层的输出
        laterals = [
            lateral_conv(inputs[i + self.start_level])  # 通过 lateral 卷积对每个输入特征图进行处理
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 构建自顶向下的路径
        used_backbone_levels = len(laterals)  # 使用的骨干网络层数，即 lateral 卷积层的数量
        for i in range(used_backbone_levels - 1, 0, -1):  # 从最后一层往前遍历
            prev_shape = laterals[i - 1].shape[2:]  # 获取上一层的空间维度（不包括 batch 和通道）
            laterals[i - 1] = self.galas[i-1](F.interpolate(  laterals[i], size=prev_shape, mode='nearest'  ), laterals[i - 1])

        # 对每一层的特征图进行 FPN 卷积处理
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        
        # 返回每一层的输出特征图
        return tuple(outs)
