import math
import warnings
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

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
#        import pdb;pdb.set_trace()
#        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.backbone_end_level = self.num_ins
        self.start_level = 0
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # Learnable weights for Fast Normalized Fusion
        self.fusion_weights = nn.Parameter(torch.ones(2*(self.num_ins-1), requires_grad=True))
        self.pre_weights = nn.Parameter(torch.ones(2*(self.num_ins)+1, requires_grad=True))

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        self.bi_convs = nn.ModuleList()
        for i in range(len(self.lateral_convs)-1):
            bi_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )
            self.bi_convs.append(bi_conv)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
    def forward(self, inputs):
        """
        Args:
            inputs (List[torch.Tensor]): Input feature maps.
              Example of shapes:
                ([1, 64, 80, 200], [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]).
        Returns:
            outputs (Tuple[torch.Tensor]): Output feature maps.
              Example of shapes:
                ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        """
        if type(inputs) == tuple:
            inputs = list(inputs)

        assert len(inputs) >= len(self.in_channels)

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Normalize weights for Fast Normalized Fusion
        pre_weights = F.relu(self.pre_weights)  # Ensure non-negative

        # build top-down path with Fast Normalized Fusion
        used_backbone_levels = len(laterals)

        middle_feature = laterals[1].clone()
        
        for i in range(used_backbone_levels - 1):
            downsampled = self.max_pool(laterals[i])
            division = pre_weights[i*2]+pre_weights[i*2+1]+1e-6

            laterals[i+1] = (
                pre_weights[i*2] /division* laterals[i+1]
                + pre_weights[i*2+1] /division* downsampled
            )
            laterals[i+1] = self.fpn_convs[i](laterals[i+1])


        # Normalize weights for Fast Normalized Fusion
        fusion_weights = F.relu(self.fusion_weights)  # Ensure non-negative

        
        laterals[-1] = self.fpn_convs[-1](laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )
            if i-1 == 1:
                division = fusion_weights[(i - 1)*2]+fusion_weights[(i - 1)*2+1]+1e-6+pre_weights[-1]
                # Apply normalized weights
                laterals[i - 1] = (
                    fusion_weights[(i - 1)*2] /division* laterals[i - 1]
                    + fusion_weights[(i - 1)*2+1] /division* upsampled+pre_weights[-1] /division* middle_feature
                )                
            else:    
                division = fusion_weights[(i - 1)*2]+fusion_weights[(i - 1)*2+1]+1e-6
                # Apply normalized weights
                laterals[i - 1] = (
                    fusion_weights[(i - 1)*2] /division* laterals[i - 1]
                    + fusion_weights[(i - 1)*2+1] /division* upsampled
                )
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])

        # Apply fpn_convs to each lateral
        
        return tuple(laterals)
