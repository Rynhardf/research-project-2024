import os
import pandas as pd
import cv2
import numpy as np
import tqdm


# Expand box by 2px on each side
def expand_box(x, y, w, h, img_width, img_height, padding=2):
    x_new = max(x - padding, 0)
    y_new = max(y - padding, 0)
    w_new = min(w + 2 * padding, img_width - x_new)
    h_new = min(h + 2 * padding, img_height - y_new)
    return x_new, y_new, w_new, h_new


# Adjust keypoints based on the crop
def adjust_keypoints(row, box_x, box_y, box_w, box_h, img_width, img_height):
    keypoints = []
    for i in range(10, len(row), 2):  # Loop through keypoints
        u = row.iloc[i]
        v = row.iloc[i + 1]

        # If keypoint is missing, continue
        if pd.isna(u) or pd.isna(v):
            keypoints.extend(["", ""])
            continue

        u = float(u)
        v = float(v)

        # Check if the keypoint is outside the original image dimensions
        if u < 0 or v < 0 or u >= img_width or v >= img_height:
            keypoints.extend(["", ""])
        else:
            # Adjust keypoints based on the cropped box
            if box_x <= u <= (box_x + box_w) and box_y <= v <= (box_y + box_h):
                keypoints.extend([u - box_x, v - box_y])
            else:
                keypoints.extend(["", ""])

    return keypoints


# Load CSV file
root_img_dir = "/mnt/e/data_processed"
csv_path = "../data/detect.csv"
df = pd.read_csv(csv_path)

# Directory to save cropped images
output_img_dir = "../data/cropped_images"
if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

num_skipped = 0

new_rows = []

for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    img_path = row["img_path"]

    # Load the image
    img = cv2.imread(os.path.join(root_img_dir, img_path))
    if img is None:
        print(f"Image {img_path} not found.")
        continue

    img_height, img_width = img.shape[:2]

    # Get bounding box details
    box_x, box_y, box_w, box_h = (
        float(row["box_x"]),
        float(row["box_y"]),
        float(row["box_w"]),
        float(row["box_h"]),
    )

    # Expand the bounding box by 2px
    box_x, box_y, box_w, box_h = expand_box(
        box_x, box_y, box_w, box_h, img_width, img_height
    )

    # Crop the image
    cropped_img = img[int(box_y) : int(box_y + box_h), int(box_x) : int(box_x + box_w)]

    # Adjust keypoints based on the new crop
    keypoints = adjust_keypoints(row, box_x, box_y, box_w, box_h, img_width, img_height)

    # Remove rows where all keypoints are outside the crop (i.e., all keypoints are empty)
    if all(kp == "" for kp in keypoints):
        # print(f"Skipping image {img_path} due to all keypoints outside the box.")
        num_skipped += 1
        continue

    # Save the cropped image
    cropped_img_name = f"cropped_{os.path.basename(img_path)}"
    cv2.imwrite(os.path.join(output_img_dir, cropped_img_name), cropped_img)

    # Create new row with the cropped image path and updated keypoints
    new_row = row[:10].tolist() + [cropped_img_name] + keypoints
    new_rows.append(new_row)

# Create new DataFrame without the box_x, box_y, box_w, box_h columns
new_columns = df.columns[:10].tolist() + ["cropped_img_path"] + df.columns[10:].tolist()
new_df = pd.DataFrame(new_rows, columns=new_columns)

# Save the new CSV file
output_csv_path = "../data/cropped.csv"
new_df.to_csv(output_csv_path, index=False)

print(f"Processed {len(new_rows)} images and saved to {output_csv_path}.")
print(f"Skipped {num_skipped} images due to all keypoints outside the box.")
