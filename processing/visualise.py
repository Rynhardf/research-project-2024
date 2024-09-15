import cv2
import csv
import numpy as np
import os

# root_path = "/mnt/e/data_processed"
root_path = "../data/cropped_images"


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
