from ultralytics import YOLO
import torch
import torch.nn as nn


class YOLOModel(nn.Module):
    def __init__(self, variant):
        super(YOLOModel, self).__init__()
        self.model = YOLO("./weights/" + variant + ".pt").model

        # unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.model(x)

        if self.model.training:
            result = out[1]
        else:
            result = out[1][1]

        result = self.model.model[22].kpts_decode(x[0], result)

        return result

    # def kpts_decode(self, bs, kpts):
    #     """Decodes keypoints."""
    #     ndim = 3

    #     y = kpts.clone()
    #     if ndim == 3:
    #         y[:, 2::3] = y[
    #             :, 2::3
    #         ].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
    #     y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
    #     y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
    #     return y


def get_yolo_model(variant, num_joints=17):
    model = YOLOModel(variant)

    return model


def output_to_scales(output, scales_sizes=[80, 40, 20]):
    # output is a (bs, 56, number of anchors)

    reshaped_output = output.reshape(output.shape[0], 17, 3, -1)

    scale_output = [
        reshaped_output[:, :, :, : scales_sizes[0] ** 2],
        reshaped_output[
            :, :, :, scales_sizes[0] ** 2 : scales_sizes[0] ** 2 + scales_sizes[1] ** 2
        ],
        reshaped_output[
            :,
            :,
            :,
            scales_sizes[0] ** 2
            + scales_sizes[1] ** 2 : scales_sizes[0] ** 2
            + scales_sizes[1] ** 2
            + scales_sizes[2] ** 2,
        ],
    ]
    if len(scales_sizes) == 4:
        scale_output.append(
            reshaped_output[
                :,
                :,
                :,
                scales_sizes[0] ** 2 + scales_sizes[1] ** 2 + scales_sizes[2] ** 2 :,
            ],
        )

    return scale_output


def get_keypoints_from_scale(scale, scale_size=80, input_size=(640, 640)):
    # scale is (bs, 17, 3, 1600)
    max_value, indices = torch.max(scale[:, :, 2, :], dim=2)

    indices = indices.unsqueeze(-1)  # Shape becomes (1, 17, 1)

    x_values = torch.gather(scale[:, :, 0, :], dim=2, index=indices)
    y_values = torch.gather(scale[:, :, 1, :], dim=2, index=indices)
    conf = torch.gather(scale[:, :, 2, :], dim=2, index=indices)

    x_values = x_values.squeeze(-1)
    y_values = y_values.squeeze(-1)
    conf = conf.squeeze(-1)

    keypoints = torch.stack([x_values, y_values, conf], dim=2)

    return keypoints


def get_anc_index(gt_keypoints, scale_size=80, input_size=(640, 640)):
    cell_size_x = input_size[0] / scale_size
    cell_size_y = input_size[1] / scale_size

    anc_x = gt_keypoints[:, :, 0] // cell_size_x
    anc_y = gt_keypoints[:, :, 1] // cell_size_y

    # return as one index out of the flattened 1600
    idx = anc_x * scale_size + anc_y

    return idx.to(torch.int64)


class YoloKeypointLoss(nn.Module):
    def __init__(self, scale_sizes=[80, 40, 20], image_size=(640, 640)):
        super(YoloKeypointLoss, self).__init__()
        self.scale_sizes = scale_sizes
        self.image_size = image_size

    def forward(self, output, target, gt_keypoints, keypoint_visibility):
        # Convert output to different scales
        scales = output_to_scales(output)

        # Combine keypoint visibility with gt_keypoints
        gt_keypoints_with_visibility = torch.cat(
            [gt_keypoints, keypoint_visibility.unsqueeze(-1)], dim=2
        )

        loss = 0
        for scale, scale_size in zip(scales, self.scale_sizes):
            idx = get_anc_index(
                gt_keypoints_with_visibility,
                scale_size=scale_size,
                input_size=self.image_size,
            )

            idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, -1)

            result = torch.gather(scale, dim=3, index=idx).squeeze(
                -1
            )  # Shape: [batch_size, 17, 3]

            # Compute confidence loss (binary cross-entropy)
            conf_loss = torch.nn.functional.binary_cross_entropy(
                result[:, :, 2], gt_keypoints_with_visibility[:, :, 2]
            )

            # Mask out invisible keypoints for location loss (MSE)
            visible_mask = (
                keypoint_visibility > 0
            )  # Boolean mask where visible keypoints are True
            visible_mask = visible_mask.unsqueeze(-1).expand(
                -1, -1, 2
            )  # Expand for x, y

            # Apply mask to keypoints and predicted results
            pred_keypoints = (
                result[:, :, :2] * visible_mask
            )  # Only consider visible keypoints
            gt_keypoints_visible = (
                gt_keypoints[:, :, :2] * visible_mask
            )  # Only visible gt keypoints

            # Compute location loss (MSE) only for visible keypoints
            loc_loss = torch.nn.functional.mse_loss(
                pred_keypoints, gt_keypoints_visible, reduction="sum"
            ) / (
                visible_mask.sum() + 1e-6
            )  # Normalize by the number of visible keypoints to avoid division by zero

            loss += conf_loss + loc_loss

        return loss


def get_keypoints_yolo(output, scale_sizes=[80, 40, 20], image_size=(640, 640)):
    scales = output_to_scales(output.clone().detach(), scale_sizes)

    keypoints = []
    for scale, scale_size in zip(scales, scale_sizes):
        keypoints.append(get_keypoints_from_scale(scale, scale_size, image_size))

    keypoints = torch.stack(keypoints, dim=1)

    confidences = keypoints[
        ..., 2
    ]  # Shape: [batch_size, number_of_scales, number_of_keypoints]

    # Get the indices of the max confidence across scales
    max_conf_idx = torch.argmax(
        confidences, dim=1
    )  # Shape: [batch_size, number_of_keypoints]

    best_keypoints = torch.zeros((keypoints.shape[0], keypoints.shape[2], 3))

    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[2]):
            best_keypoints[i, j] = keypoints[i, max_conf_idx[i, j], j]

    confidences = best_keypoints[:, :, 2]
    best_keypoints = best_keypoints[:, :, :2]

    return best_keypoints
