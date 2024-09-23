from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


# blocks
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Pose(nn.Module):

    def __init__(self, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super(Pose, self).__init__()
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.nl = len(ch)  # number of detection layers

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))
            for x in ch
        )

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1
        )  # (bs, 17*3, h*w)
        return kpt


class YOLO(nn.Module):
    def __init__(
        self,
        width_mult=1.25,
        depth_mult=1.0,
        resolution_mult=1.0,
        num_keypoints=17,
        strides_sizes=[8, 16, 32],
        img_size=(640, 640),
    ):
        super(YOLO, self).__init__()

        # Scaling factors
        w = width_mult
        d = depth_mult
        r = resolution_mult

        self.strides_sizes = strides_sizes
        self.img_size = img_size

        self.init_stride_anchors()

        self.model = nn.ModuleList(
            [
                # Backbone part
                Conv(3, int(64 * w), k=3, s=2, p=1),  # Layer 0: Conv
                Conv(int(64 * w), int(128 * w), k=3, s=2, p=1),  # Layer 1: Conv
                C2f(
                    int(128 * w), int(128 * w), n=int(3 * d), shortcut=True
                ),  # Layer 2: C2f
                Conv(int(128 * w), int(256 * w), k=3, s=2, p=1),  # Layer 3: Conv
                C2f(
                    int(256 * w), int(256 * w), n=int(6 * d), shortcut=True
                ),  # Layer 4: C2f
                Conv(int(256 * w), int(512 * w), k=3, s=2, p=1),  # Layer 5: Conv
                C2f(
                    int(512 * w), int(512 * w), n=int(6 * d), shortcut=True
                ),  # Layer 6: C2f
                Conv(int(512 * w), int(512 * w * r), k=3, s=2, p=1),  # Layer 7: Conv
                C2f(
                    int(512 * w), int(512 * w * r), n=int(3 * d), shortcut=True
                ),  # Layer 8: C2f
                SPPF(int(512 * w * r), int(512 * w * r), k=5),  # Layer 9: SPPF
                # Neck part
                nn.Upsample(scale_factor=2.0, mode="nearest"),  # Layer 10: Upsample
                Concat(),  # Layer 11: Concat (skip connection)
                C2f(
                    int(512 * w * (1 + r)), int(512 * w), n=int(3 * d), shortcut=False
                ),  # Layer 12: C2f
                nn.Upsample(scale_factor=2.0, mode="nearest"),  # Layer 13: Upsample
                Concat(),  # Layer 14: Concat (skip connection)
                C2f(
                    int(768 * w), int(256 * w), n=int(3 * d), shortcut=False
                ),  # Layer 15: C2f
                Conv(int(256 * w), int(256 * w), k=3, s=2, p=1),  # Layer 16: Conv
                Concat(),  # Layer 17: Concat
                C2f(
                    int(768 * w), int(512 * w), n=int(3 * d), shortcut=False
                ),  # Layer 18: C2f
                Conv(int(512 * w), int(512 * w), k=3, s=2, p=1),  # Layer 19: Conv
                Concat(),  # Layer 20: Concat
                C2f(
                    int(512 * w * (1 + r)),
                    int(512 * w * r),
                    n=int(3 * d),
                    shortcut=False,
                ),  # Layer 21: C2f
                Pose(
                    kpt_shape=(17, 3), ch=[int(256 * w), int(512 * w), int(512 * w * r)]
                ),
            ]
        )

    def init_stride_anchors(self):
        self.grid_sizes = [self.img_size[0] // s for s in self.strides_sizes]
        strides_array = []
        for stride, grid_size in zip(self.strides_sizes, self.grid_sizes):
            strides_array.extend([stride] * (grid_size**2))
        self.strides = torch.tensor(strides_array, dtype=torch.float32).unsqueeze(0)

        anchors_x = []
        anchors_y = []

        for grid_size in self.grid_sizes:
            x_values = np.arange(0.5, grid_size, 1.0)
            y_values = np.arange(0.5, grid_size, 1.0)

            x, y = np.meshgrid(x_values, y_values)

            anchors_x.append(x.reshape(-1))
            anchors_y.append(y.reshape(-1))

        self.anchors = torch.tensor(
            np.stack([np.concatenate(anchors_x), np.concatenate(anchors_y)], axis=1),
            dtype=torch.float32,
        ).permute(1, 0)

    def forward(self, x):
        # implement YOLOv8 forward pass

        # Backbone part
        x0 = self.model[0](x)
        x1 = self.model[1](x0)
        x2 = self.model[2](x1)
        x3 = self.model[3](x2)
        x4 = self.model[4](x3)
        x5 = self.model[5](x4)
        x6 = self.model[6](x5)
        x7 = self.model[7](x6)
        x8 = self.model[8](x7)
        x9 = self.model[9](x8)

        # Neck part
        x10 = self.model[10](x9)
        x11 = self.model[11]([x10, x6])
        x12 = self.model[12](x11)
        x13 = self.model[13](x12)
        x14 = self.model[14]([x13, x4])
        x15 = self.model[15](x14)
        x16 = self.model[16](x15)
        x17 = self.model[17]([x16, x12])
        x18 = self.model[18](x17)
        x19 = self.model[19](x18)
        x20 = self.model[20]([x19, x9])
        x21 = self.model[21](x20)

        x = [x15, x18, x21]

        result = self.model[22](x)

        result = self.decode_kpt(result)

        return result

    def decode_kpt(self, output):
        self.anchors = self.anchors.to(output.device)
        self.strides = self.strides.to(output.device)

        output[:, 2::3, :] = torch.sigmoid(output[:, 2::3, :])
        output[:, 0::3, :] = (
            output[:, 0::3, :] * 2.0 + (self.anchors[0] - 0.5)
        ) * self.strides
        output[:, 1::3, :] = (
            output[:, 1::3, :] * 2.0 + (self.anchors[1] - 0.5)
        ) * self.strides

        return output


def get_keypoints_yolo(output):
    conf = output[:, 2::3, :]
    x_values = output[:, 0::3, :]
    y_values = output[:, 1::3, :]

    idx = torch.argmax(conf, dim=2)

    x_values = torch.gather(x_values, 2, idx.unsqueeze(2))
    y_values = torch.gather(y_values, 2, idx.unsqueeze(2))
    conf = torch.gather(conf, 2, idx.unsqueeze(2))

    keypoints = torch.cat((x_values, y_values), dim=2)

    return keypoints


yolo_models = {
    "yolov8x-pose": {
        "d": 1.0,
        "w": 1.25,
        "r": 1.0,
        "strides_sizes": [8, 16, 32],
        "img_size": (640, 640),
    }
}


def get_yolo_model(variant, num_joints=17):
    params = yolo_models[variant]
    model = YOLO(
        width_mult=params["w"],
        depth_mult=params["d"],
        resolution_mult=params["r"],
        num_keypoints=num_joints,
        strides_sizes=params["strides_sizes"],
        img_size=params["img_size"],
    )

    loaded_weights = torch.load("./weights/" + variant + ".pt")

    model.load_state_dict(loaded_weights["model"].state_dict(), strict=False)

    return model


class YoloKeypointLoss(nn.Module):
    def __init__(self, strides_sizes=[8, 16, 32], image_size=640):
        super(YoloKeypointLoss, self).__init__()
        self.strides_sizes = strides_sizes
        self.image_size = image_size

    def forward(self, output, target, gt_keypoints, keypoint_visibility):
        conf = output[:, 2::3, :]
        x_values = output[:, 0::3, :]
        y_values = output[:, 1::3, :]

        bs = output.size(0)
        num_kp = output.size(1) // 3

        stride_idxs = []

        for stride in self.strides_sizes:
            grid_size = self.image_size // stride
            block_x = gt_keypoints[:, :, 0] // stride
            block_y = gt_keypoints[:, :, 1] // stride
            stride_idxs.append(block_y * grid_size + block_x)

        stride_idxs = torch.stack(stride_idxs, dim=2).long().to(output.device)

        conf_mask = torch.zeros_like(conf)
        x_loss = 0
        y_loss = 0
        for i in range(bs):
            for j in range(num_kp):
                if keypoint_visibility[i, j] == 1:
                    for k in range(1):
                        idx = stride_idxs[i, j, k]
                        conf_mask[i, j, idx] = 1
                        x_loss += nn.functional.mse_loss(
                            x_values[i, j, idx], gt_keypoints[i, j, 0]
                        )
                        y_loss += nn.functional.mse_loss(
                            y_values[i, j, idx], gt_keypoints[i, j, 1]
                        )

        conf_loss = nn.functional.binary_cross_entropy(
            conf, conf_mask, reduction="mean"
        )

        x_loss /= bs
        y_loss /= bs

        return conf_loss + x_loss + y_loss
