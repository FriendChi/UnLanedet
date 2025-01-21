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

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块
class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1
class EMA(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=32):  # 构造函数，初始化对象
        super(EMA, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 定义自适应平均池化层，只在高度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 定义 1x1 卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # 定义 3x3 卷积层
 
    def forward(self, x):  # 定义前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)
        x_h = self.pool_h(group_x)  # 在高度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 在宽度上进行池化并交换维度
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化结果拼接并通过 1x1 卷积层
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积结果按高度和宽度分割
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 进行组归一化，并结合高度和宽度的激活结果
        x2 = self.conv3x3(group_x)  # 通过 3x3 卷积层
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 应用权重并将形状恢复为原始大小


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
        self.se_list = nn.ModuleList()
        self.mid_module = EMA(64)
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
            self.se_list.append(EMA(out_channels))
            
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
        inputs[0] =  self.mid_module(inputs[0])
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
            laterals[i - 1] += F.interpolate(  # 将当前层的特征图上采样到上一层的大小并进行融合
                laterals[i], size=prev_shape, mode='nearest'  # 使用最近邻插值方法进行上采样
            )

        # 对每一层的特征图进行 FPN 卷积处理
        outs = [self.se_list[i](laterals[i]) for i in range(used_backbone_levels)]
        
        # 返回每一层的输出特征图
        return tuple(outs)
