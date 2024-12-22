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
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


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
        self.ca_list = nn.ModuleList()
        
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
            self.ca_list.append(CoordAtt(in_channels[i],in_channels[i]))
            self.lateral_convs.append(l_conv)  # 将 lateral 卷积层添加到列表中
            self.fpn_convs.append(fpn_conv)  # 将 FPN 卷积层添加到列表中

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
            lateral_conv(self.ca_list[i](inputs[i + self.start_level]))  # 通过 lateral 卷积对每个输入特征图进行处理
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 构建自顶向下的路径
        used_backbone_levels = len(laterals)  # 使用的骨干网络层数，即 lateral 卷积层的数量
        for i in range(used_backbone_levels - 1, 0, -1):  # 从最后一层往前遍历
            prev_shape = laterals[i - 1].shape[2:]  # 获取上一层的空间维度（不包括 batch 和通道）
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest'  )
            
            
        # 对每一层的特征图进行 FPN 卷积处理
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        
        # 返回每一层的输出特征图
        return tuple(outs)
