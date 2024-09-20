from ultralytics import YOLO
import torch


def get_yolo_model(variant):
    model = YOLO("./weights/" + variant + ".pt")

    return model.model


def output_to_scales(output, scales=[80, 40, 20]):
    # output is a (bs, 56, number of anchors)

    reshaped_output = output[:, 5:, :].reshape(output.shape[0], 17, 3, -1)

    scale_output = [
        reshaped_output[:, :, :, : scales[0] ** 2],
        reshaped_output[:, :, :, scales[0] ** 2 : scales[0] ** 2 + scales[1] ** 2],
        reshaped_output[
            :,
            :,
            :,
            scales[0] ** 2
            + scales[1] ** 2 : scales[0] ** 2
            + scales[1] ** 2
            + scales[2] ** 2,
        ],
    ]
    if len(scales) == 4:
        scale_output.append(
            reshaped_output[
                :, :, :, scales[0] ** 2 + scales[1] ** 2 + scales[2] ** 2 :
            ],
        )

    return scale_output


def get_keypoints_from_scale(scale, scale_size=80, input_size=(640, 640)):
    # scale is (bs, 17, 3, 1600)
    max_value, indices = torch.max(scale[:, :, 2, :], dim=2)

    indices = indices.unsqueeze(-1)  # Shape becomes (1, 17, 1)

    x_values = torch.gather(scale[:, :, 0, :], dim=2, index=indices)
    y_values = torch.gather(scale[:, :, 1, :], dim=2, index=indices)

    x_values = x_values.squeeze(-1)
    y_values = y_values.squeeze(-1)

    keypoints = torch.stack([x_values, y_values], dim=2)

    return keypoints
