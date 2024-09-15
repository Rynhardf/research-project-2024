import csv
import cv2
import os
import numpy as np
import json
import tqdm
import random

# Configurable percentages for splits
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# Check that the percentages sum to 1.0
assert (
    abs(train_percent + val_percent + test_percent - 1.0) < 1e-6
), "Percentages must sum to 1.0"

# Define which keypoints to include and in what order
selected_keypoints = [
    "C7",
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
    "LSHO",
    "LFIN",
    "LEJC",
    "REJC",
    "LWJC",
    "RWJC",
    "RHEE",
    "RTOE",
    "LKJC",
    "RKJC",
    "LAJC",
    "RAJC",
]

# All keypoints present in the data
all_keypoints = [
    "C7",
    "CLAV",
    "CentreOfMass",
    "CentreOfMassFloor",
    "LAJC",
    "LANK",
    "LASI",
    "LBHD",
    "LEJC",
    "LELB",
    "LFHD",
    "LFIN",
    "LFRM",
    "LHEE",
    "LHJC",
    "LKJC",
    "LKNE",
    "LMMED",
    "LPSI",
    "LSHO",
    "LSJC",
    "LTHI",
    "LTIB",
    "LTOE",
    "LUPA",
    "LWJC",
    "LWRA",
    "LWRB",
    "L_Foot_Out",
    "MElbowL",
    "MElbowR",
    "MKNEL",
    "MKNER",
    "PelL",
    "PelR",
    "RAJC",
    "RANK",
    "RASI",
    "RBAK",
    "RBHD",
    "REJC",
    "RELB",
    "RFHD",
    "RFIN",
    "RFRM",
    "RHEE",
    "RHJC",
    "RKJC",
    "RKNE",
    "RMMED",
    "RPSI",
    "RSHO",
    "RSJC",
    "RTHI",
    "RTIB",
    "RTOE",
    "RUPA",
    "RWJC",
    "RWRA",
    "RWRB",
    "R_Foot_Out",
    "STRN",
    "T10",
]

# Mapping from all keypoints to their indices in the data
keypoint_indices = []
for kp in selected_keypoints:
    keypoint_indices.append(all_keypoints.index(kp))

extend_bbox = 2  # Extend bounding box by 2 pixels on each side
root_path_save = "coco"
root_path_img = "/mnt/e/data_processed"

csv_file = "detect.csv"


def filter_keypoints(keypoints):
    return keypoints[keypoint_indices]


def get_keypoints(keypoints_str, img_width, img_height):
    keypoints = []
    for i in range(0, len(keypoints_str), 2):
        x = keypoints_str[i]
        y = keypoints_str[i + 1]
        if x == "" or y == "":
            keypoints.extend([0, 0, 0])
            continue
        x = float(x)
        y = float(y)
        if x < 0 or x > img_width or y < 0 or y > img_height:
            keypoints.extend([0, 0, 0])
        else:
            keypoints.extend([x, y, 2])

    keypoints = np.array(keypoints)
    keypoints = keypoints.reshape(-1, 3)
    return keypoints


# Function to adjust keypoints based on the new cropped bounding box
def adjust_keypoints(keypoints, box_x, box_y):
    adjusted_keypoints = keypoints.copy()
    for i in range(len(adjusted_keypoints)):
        adjusted_keypoints[i, 0] = adjusted_keypoints[i, 0] - box_x
        adjusted_keypoints[i, 1] = adjusted_keypoints[i, 1] - box_y
    return adjusted_keypoints


# Function to check if keypoints are within the bounding box
def keypoints_within_box(keypoints, box_x, box_y, box_w, box_h):
    for i, kp in enumerate(keypoints):
        x = kp[0]
        y = kp[1]
        v = kp[2]

        if v == 0:
            continue

        if x < box_x or x > box_x + box_w or y < box_y or y > box_y + box_h:
            return False
    return True


# Read the CSV file
with open(csv_file, newline="") as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    rows = list(csvreader)

# Shuffle the data
random.shuffle(rows)

# Split the data
num_rows = len(rows)
train_end = int(num_rows * train_percent)
val_end = train_end + int(num_rows * val_percent)

train_rows = rows[:train_end]
val_rows = rows[train_end:val_end]
test_rows = rows[val_end:]

splits = [("train", train_rows), ("val", val_rows), ("test", test_rows)]

for split_name, split_rows in splits:
    print(f"Processing {split_name} data with {len(split_rows)} samples.")
    # Prepare output directories for images and annotations
    output_image_dir = f"./{split_name}_images/"
    os.makedirs(os.path.join(root_path_save, output_image_dir), exist_ok=True)

    # COCO annotation template for each split
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": selected_keypoints,  # Use only selected keypoints
                "num_keypoints": len(selected_keypoints),
                "skeleton": [],  # Define skeleton connections if needed
            }
        ],
    }

    annotation_id = 1
    image_id = 1
    num_outside_bbox = 0

    for row in tqdm.tqdm(split_rows, total=len(split_rows)):
        C, S, A, D, frame_num = row[:5]
        img_path = row[5]
        box_x, box_y, box_w, box_h = map(float, row[6:10])

        image_full_path = os.path.join(root_path_img, img_path)
        image = cv2.imread(image_full_path)

        if image is None:
            print(f"Image {image_full_path} not found.")
            continue

        keypoints = get_keypoints(row[10:], image.shape[1], image.shape[0])
        keypoints = filter_keypoints(keypoints)

        # Extend the bounding box by 2 pixels on each side
        box_x = max(0, box_x - extend_bbox)
        box_y = max(0, box_y - extend_bbox)
        box_w = box_w + extend_bbox * 2
        box_h = box_h + extend_bbox * 2

        # Check if keypoints are within the bounding box
        if not keypoints_within_box(keypoints, box_x, box_y, box_w, box_h):
            # print(f"Skipping image {img_path} due to keypoints outside the bounding box")
            num_outside_bbox += 1
            continue

        # Crop the image using the extended bounding box
        cropped_image = image[
            int(box_y) : int(box_y + box_h), int(box_x) : int(box_x + box_w)
        ]

        # Adjust the keypoints relative to the new cropped image
        adjusted_keypoints = adjust_keypoints(keypoints, box_x, box_y)

        # Save the cropped image
        cropped_image_path = os.path.join(
            output_image_dir, f"C{C}S{S}A{A}D{D}_{frame_num}.jpg"
        )
        cv2.imwrite(os.path.join(root_path_save, cropped_image_path), cropped_image)

        # Prepare COCO image and annotation data
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": cropped_image_path,
                "width": cropped_image.shape[1],
                "height": cropped_image.shape[0],
            }
        )

        coco_data["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": adjusted_keypoints.flatten().tolist(),
                "num_keypoints": int(np.count_nonzero(adjusted_keypoints[:, 2] > 0)),
                "bbox": [0, 0, cropped_image.shape[1], cropped_image.shape[0]],
                "iscrowd": 0,
            }
        )

        annotation_id += 1
        image_id += 1

    # Save the COCO annotations to a JSON file
    annotations_file = os.path.join(
        root_path_save, f"coco_annotations_{split_name}.json"
    )
    with open(annotations_file, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(
        f"Number of images outside the bounding box in {split_name} set: {num_outside_bbox}"
    )
    print(
        f"COCO format data and cropped images for {split_name} set saved successfully."
    )
