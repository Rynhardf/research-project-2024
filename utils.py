import torch
import torch.nn as nn
import yaml
import cv2
import numpy as np
import os


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, save_path):
    with open(save_path, "w") as file:
        yaml.dump(config, file)


def recursive_freeze(module, layer_config):
    for layer_name, freeze_or_subconfig in layer_config.items():
        if isinstance(freeze_or_subconfig, bool):
            if freeze_or_subconfig and hasattr(module, layer_name):
                for param in getattr(module, layer_name).parameters():
                    param.requires_grad = False
        elif isinstance(freeze_or_subconfig, dict):
            if hasattr(module, layer_name):
                submodule = getattr(module, layer_name)
                recursive_freeze(submodule, freeze_or_subconfig)


def load_model(config):
    if config["name"] == "HRNet":
        from models.HRNet.hrnet import get_pose_model

        model = get_pose_model(
            config["W"],
            config["num_joints"],
        )
        
        if config["weights"]:
            model.init_weights(config["weights"])
    elif config["name"] == "ViTPose":
        from models.ViTPose.vitpose import get_vitpose_model

        model = get_vitpose_model(
            variant=config["variant"],
            num_joints=config["num_joints"],
            decoder=config["decoder"],
        )

        if config["weights"]:
            weights = torch.load(config["weights"])
            if "state_dict" in weights.keys():
                weights = weights["state_dict"]
            model.init_weights(weights)
    else:
        raise ValueError(
            f"Model {config['model']['name']} not recognized"
        )  # Added error handling

    if "freeze" in config.keys():
        recursive_freeze(model, config["freeze"])

    return model


class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, output, target, keypoint_visibility):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        loss = 0
        keypoint_visibility = keypoint_visibility.unsqueeze(-1)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[:, idx]
            heatmap_gt = heatmaps_gt[:, idx]

            visible_pred = heatmap_pred * keypoint_visibility[:, idx]
            visible_gt = heatmap_gt * keypoint_visibility[:, idx]

            joint_loss = self.criterion(visible_pred, visible_gt)
            loss += 0.5 * joint_loss

        return loss / num_joints


# Function to draw keypoints on an image
def draw_keypoints(image_path, keypoints, box, C, S, A, D, frame_num):
    # Load the image
    image = cv2.imread(os.path.join(root_path, image_path))

    if box is None:
        box = (0, 0, image.shape[1], image.shape[0])

    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Iterate through keypoints in pairs (x, y)
    for i in range(0, len(keypoints), 2):
        x = keypoints[i]
        y = keypoints[i + 1]

        # Only draw if x and y are not NaN
        if not np.isnan(x) and not np.isnan(y):
            # Draw a circle for each keypoint
            cv2.circle(
                image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1
            )

    # Put the text for CSAD and frame_num on the image
    text = f"C: {C}, S: {S}, A: {A}, D: {D}, Frame: {frame_num}"

    cv2.putText(
        image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
    )

    # Draw the bounding box
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # resize the image to half
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Keypoints", image)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        exit()


def visualize_output(image, keypoints, heatmaps):
    """
    Visualize image with keypoints and heatmap using OpenCV.

    Args:
    - image: A numpy array representing the image.
    - keypoints: A list of tuples representing the (x, y) coordinates of keypoints. It should be in the format [(x1, y1), (x2, y2), ...].
    - heatmaps: A numpy array representing the heatmaps.
    """

    # sum all heatmaps
    heatmap = np.sum(heatmaps, axis=0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)

    # Check if image is None
    if image is None:
        print("Image is None, cannot visualize.")
        return

    img_height, img_width = image.shape[:2]

    # Resize the heatmap to match the image dimensions
    if heatmap is not None:
        # Normalize the heatmap for visualization (scale to 0-255)
        heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
        heatmap_resized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_resized = heatmap_resized.astype(np.uint8)

        # Convert to color heatmap
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Overlay the heatmap on the image with some transparency
        overlayed_image = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    else:
        overlayed_image = image

    # Draw keypoints if they are provided
    if keypoints is not None:
        for point in keypoints:
            x, y = int(point[0]), int(point[1])

            # Draw a circle for each keypoint on the image
            cv2.circle(
                overlayed_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1
            )

    # Show the result in a window
    cv2.imshow("Visualization", overlayed_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def get_keypoints_from_heatmaps(heatmaps, image_size):
    """
    Convert predicted heatmaps to keypoint coordinates using PyTorch.

    Args:
    - heatmaps (torch.Tensor): Heatmaps of shape (batch_size, num_keypoints, heatmap_height, heatmap_width)
    - image_size (tuple): The original image size (image_height, image_width)

    Returns:
    - keypoints (torch.Tensor): Predicted keypoint locations of shape (batch_size, num_keypoints, 2),
                                where each keypoint is represented by (x, y) coordinates.
    """
    # Ensure the heatmaps are torch tensors (if not already)
    if not isinstance(heatmaps, torch.Tensor):
        raise TypeError("Heatmaps should be a torch.Tensor")

    batch_size, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape
    img_height, img_width = image_size

    # Flatten the heatmap to find the maximum value positions
    heatmaps_reshaped = heatmaps.reshape(
        batch_size, num_keypoints, -1
    )  # Flatten height and width into one dimension
    max_vals, max_indices = torch.max(
        heatmaps_reshaped, dim=-1
    )  # Find max value and corresponding index in flattened heatmap

    # Convert the 1D indices back to 2D coordinates
    max_indices_x = max_indices % heatmap_width
    max_indices_y = max_indices // heatmap_width

    # Scale the coordinates to match the original image size
    max_indices_x = (max_indices_x.float() / heatmap_width) * img_width
    max_indices_y = (max_indices_y.float() / heatmap_height) * img_height

    # Stack the x and y coordinates
    keypoints = torch.stack(
        [max_indices_x, max_indices_y], dim=-1
    )  # Shape: (batch_size, num_keypoints, 2)

    return keypoints


def inference(config, images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config["model"])
    model = model.to(device)
    images = images.to(device)
    # Inference
    model.eval()
    with torch.no_grad():
        output_batch = model(images)  # Get outputs for all images in the batch

    return output_batch.cpu()


def normalized_mae_in_pixels(predictions, targets, image_shape, keypoint_visibility):
    """
    Compute the normalized Mean Absolute Error (MAE) in pixels.
    Normalized by the diagonal of the input images, with visibility masking.

    Args:
        predictions: torch.Tensor (batch_size, num_keypoints, 2)
            Model's predicted keypoint locations (x, y).
        targets: torch.Tensor (batch_size, num_keypoints, 2)
            Ground truth keypoint locations (x, y).
        image_shape: tuple (int, int)
            Dimensions of the input image (width, height) to normalize the error.
        keypoint_visibility: torch.Tensor (batch_size, num_keypoints)
            Visibility flags for keypoints (1 for visible, 0 for not visible).

    Returns:
        float: Normalized MAE in pixels.
    """
    image_width, image_height = image_shape
    image_diagonal = torch.sqrt(
        torch.tensor(image_width**2 + image_height**2, dtype=torch.float32)
    )

    # Ensure keypoint_visibility is broadcasted to (batch_size, num_keypoints, 2)
    keypoint_visibility = keypoint_visibility.unsqueeze(-1).expand_as(predictions)

    visible_predictions = predictions * keypoint_visibility
    visible_targets = targets * keypoint_visibility

    euclidean_error = torch.sqrt(
        torch.sum((visible_predictions - visible_targets) ** 2, dim=-1)
    )

    # Sum over all keypoints and batches
    total_error = euclidean_error.sum(dim=(0, 1))
    visible_count = keypoint_visibility[..., 0].sum(
        dim=(0, 1)
    )  # Count visible keypoints

    # Avoid division by zero if there are no visible keypoints
    if visible_count > 0:
        normalized_mae = total_error / (visible_count * image_diagonal)
    else:
        normalized_mae = 0.0  # If no keypoints are visible, set MAE to 0

    return normalized_mae.item() * 100


def compute_oks(predictions, targets, keypoint_visibility, sigmas, image_area):
    """
    Compute Object Keypoint Similarity (OKS).

    Args:
        predictions: torch.Tensor (batch_size, num_keypoints, 2)
            Model's predicted keypoint locations (x, y).
        targets: torch.Tensor (batch_size, num_keypoints, 2)
            Ground truth keypoint locations (x, y).
        keypoint_visibility: torch.Tensor (batch_size, num_keypoints)
            Visibility flags for keypoints (1 for visible, 0 for not visible).
        sigmas: torch.Tensor (num_keypoints,)
            Standard deviation values per keypoint type to weight their importance.
        image_area: float
            Area of the object (bounding box area, or entire image area).

    Returns:
        float: OKS score.
    """
    # Compute the squared Euclidean distance between predicted and ground truth keypoints
    d_sq = ((predictions - targets) ** 2).sum(
        dim=-1
    )  # Shape (batch_size, num_keypoints)

    # Scale distance by the area and keypoint-specific sigma
    scaled_d_sq = d_sq / (2 * (sigmas**2) * image_area)

    # Apply keypoint visibility mask
    oks = torch.exp(-scaled_d_sq) * keypoint_visibility

    # Average OKS over all visible keypoints and batches
    visible_count = keypoint_visibility.sum(
        dim=1
    )  # Count of visible keypoints per batch
    oks_mean = (oks.sum(dim=1) / visible_count).mean()  # Mean OKS across the batch

    return oks_mean.item()  # Return as scalar
