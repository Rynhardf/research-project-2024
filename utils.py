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


def load_model(config):
    if config["name"] == "HRNet":
        from models.HRNet.hrnet import PoseHighResolutionNet

        model = PoseHighResolutionNet(config["config"])
        if config["weights"]:
            model.init_weights(config["weights"])
    else:
        raise ValueError(
            f"Model {config['model']['name']} not recognized"
        )  # Added error handling
    return model


class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, output, target):
        # # check if output or target contains NaN or Inf
        # if torch.isnan(output).any():
        #     print("Output contains NaN")
        # if torch.isnan(target).any():
        #     print("Target contains NaN")
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

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
    Convert predicted heatmaps to keypoint coordinates.

    Args:
    - heatmaps (numpy.ndarray or torch.Tensor): Heatmaps of shape (batch_size, num_keypoints, heatmap_height, heatmap_width)
    - image_size (tuple): The original image size (image_height, image_width)

    Returns:
    - keypoints (numpy.ndarray): Predicted keypoint locations of shape (batch_size, num_keypoints, 2), where each keypoint is represented by (x, y) coordinates.
    """

    # Ensure the heatmaps are NumPy arrays
    if isinstance(heatmaps, np.ndarray) is False:
        heatmaps = heatmaps.cpu().numpy()  # In case it's a torch.Tensor

    batch_size, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape
    img_height, img_width = image_size

    # Initialize an array to store the predicted keypoints
    keypoints = np.zeros((batch_size, num_keypoints, 2), dtype=np.float32)

    for i in range(batch_size):
        for j in range(num_keypoints):
            # Find the index of the maximum value in the heatmap for each keypoint
            max_pos = np.unravel_index(
                np.argmax(heatmaps[i, j]), (heatmap_height, heatmap_width)
            )
            y, x = max_pos

            # Scale the keypoint coordinates to the original image size
            x = (x / heatmap_width) * img_width
            y = (y / heatmap_height) * img_height

            # Store the result
            keypoints[i, j] = [x, y]

    return keypoints


def inference(config, images):
    model = load_model(config["model"])

    # Inference
    model.eval()
    with torch.no_grad():
        output_batch = model(images)  # Get outputs for all images in the batch

    return output_batch  # Returns the batch of outputs
