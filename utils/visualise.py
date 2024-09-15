import cv2
import csv
import numpy as np
import os

# root_path = "/mnt/e/data_processed"
root_path = "../data/cropped_images"


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
            if point is None or point[2] == 0:
                continue  # Skip empty or invalid keypoints
            x, y = int(point[0]), int(point[1])

            # Draw a circle for each keypoint on the image
            cv2.circle(
                overlayed_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1
            )

    # Show the result in a window
    cv2.imshow("Visualization", overlayed_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


# # Read the CSV file and extract relevant data
# csv_file = "../data/cropped.csv"

# n = 0

# include_box = True
# cropped = True

# with open(csv_file, newline="") as csvfile:
#     csvreader = csv.reader(csvfile)

#     # Skip the header if there is one
#     next(csvreader)

#     # Skip the first n rows
#     for _ in range(n):
#         next(csvreader)

#     # Loop through each row in the CSV
#     for row in csvreader:
#         C = row[0]
#         S = row[1]
#         A = row[2]
#         D = row[3]
#         frame_num = row[4]

#         # Image path
#         image_path = row[5]  # Assuming the 6th column contains the image path

#         if include_box:
#             box_x = float(row[6])
#             box_y = float(row[7])
#             box_w = float(row[8])
#             box_h = float(row[9])
#             box = (box_x, box_y, box_w, box_h)
#         else:
#             box = None

#         cropped_image_path = row[10]

#         # Extract keypoints (assuming from the 7th column onwards are the x, y keypoints)
#         if include_box:
#             points = row[11:]
#         else:
#             points = row[6:]

#         keypoints = []
#         for point in points:
#             if point == "":
#                 keypoints.append(np.nan)
#             else:
#                 keypoints.append(float(point))

#         # Draw and show keypoints on the image
#         path = cropped_image_path if cropped else image_path
#         if cropped:
#             box = None
#         draw_keypoints(path, keypoints, box, C, S, A, D, frame_num)
