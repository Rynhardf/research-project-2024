# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

# common params for NETWORK
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.IMAGE_SIZE = [288, 384]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [72, 96]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 3
_C.MODEL.EXTRA = CN(new_allowed=True)


# pose_multi_resoluton_net related params
HRNET = CN()
HRNET.PRETRAINED_LAYERS = ["*"]
HRNET.FINAL_CONV_KERNEL = 1

HRNET.STAGE2 = CN()
HRNET.STAGE2.NUM_MODULES = 1
HRNET.STAGE2.NUM_BRANCHES = 2
HRNET.STAGE2.NUM_BLOCKS = [4, 4]
HRNET.STAGE2.NUM_CHANNELS = [48, 96]
HRNET.STAGE2.BLOCK = "BASIC"
HRNET.STAGE2.FUSE_METHOD = "SUM"

HRNET.STAGE3 = CN()
HRNET.STAGE3.NUM_MODULES = 4
HRNET.STAGE3.NUM_BRANCHES = 3
HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET.STAGE3.NUM_CHANNELS = [48, 96, 192]
HRNET.STAGE3.BLOCK = "BASIC"
HRNET.STAGE3.FUSE_METHOD = "SUM"

HRNET.STAGE4 = CN()
HRNET.STAGE4.NUM_MODULES = 3
HRNET.STAGE4.NUM_BRANCHES = 4
HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
HRNET.STAGE4.BLOCK = "BASIC"
HRNET.STAGE4.FUSE_METHOD = "SUM"

_C.MODEL.EXTRA = HRNET
