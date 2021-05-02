# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from models.resnet import resnet50
from packages.detr.util.misc import NestedTensor, is_main_process
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .position_encoding import build_position_encoding
# from models.model_LMDT01 import FT


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class HMExtractor(nn.Module):
    def __init__(self, num_channels=1024, bnorm=False):
        super(HMExtractor, self).__init__()
        self.ft1 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, 1024, 1, 1, padding=0, dilation=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 1, 1, padding=0, dilation=6, output_padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.ft2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 1, 1, padding=0, dilation=6, output_padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, 3, 1, padding=0, dilation=6, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.ft3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 1, 1, padding=0, dilation=6, output_padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, 1, padding=0, dilation=6, output_padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.ft4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 1, padding=0, dilation=6, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 68, 3, 1, padding=0, dilation=4, output_padding=0),
            nn.BatchNorm2d(68),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        x = self.ft1(x)
        x = self.ft2(x)
        x = self.ft3(x)
        x = self.ft4(x)
        return x


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        hm = HMExtractor()
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, pretrained: bool):
        backbone = resnet50(pretrained=pretrained, progress=True,
                            replace_stride_with_dilation=[False, False, dilation], norm_layer=FrozenBatchNorm2d)
        # backbone = getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, hm_extractor=None):
        super().__init__(backbone, position_embedding, hm_extractor)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        outxs = {'0': xs.pop('3')} if xs.__len__() != 1 else xs
        hmxs = self[2](xs['2'].tensors) if (xs.__len__() != 1) and (self[2] is not None) else None
        out: List[NestedTensor] = []
        pos = []
        for name, x in outxs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos, hmxs


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.return_interm_layers
    hm_extractor = HMExtractor() if args.heatmap_regression_via_backbone else None
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.backbone_pretrained)
    model = Joiner(backbone, position_embedding, hm_extractor=hm_extractor)
    model.num_channels = backbone.num_channels
    return model
