import os
import json
import tqdm
import csv
import numpy as np

# Replace 'your_directory_path' with the path to your dataset
data_root = "/mnt/e/internal_data"

include_xyz = False

with open("./point_ids.json", "r") as f:
    data = json.load(f)
    POINT_IDS = data["point_ids"]


def get_folders(root):
    return [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]


def get_files(root):
    return [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]


def process_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    frames = []
    for frame in data:
        frame_num, img_path, xy, xyz = process_frame(frame)
        frames.append((frame_num, img_path, xy, xyz))

    return frames


def process_frame(data):
    point_ids = data["point_ids"]
    if point_ids != POINT_IDS:
        print("Point IDs do not match")
        return

    frame_num = data["frame_num"]
    img_path = data["path"]
    xy = data["xy"]
    xyz = data["xyz"]

    xy = np.array(xy).astype(np.float32)
    xyz = np.array(xyz).astype(np.float32)

    xy = np.round(xy, 2)
    xyz = np.round(xyz, 2)

    return frame_num, img_path, xy, xyz


camera_folders = get_folders(data_root)
FILES = []

for camera_folder in camera_folders:
    C = camera_folder.replace("C", "")
    subjects = get_folders(os.path.join(data_root, camera_folder))

    for subject in subjects:
        S = subject.replace("S", "")
        sequences = get_files(os.path.join(data_root, camera_folder, subject))

        for sequence in sequences:
            if not sequence.endswith(".json"):
                continue
            seq = sequence.replace(".json", "")
            A = seq[seq.index("A") + 1 : seq.index("D")]
            D = seq[seq.index("D") + 1 : len(seq)]

            file = os.path.join(data_root, camera_folder, subject, sequence)
            FILES.append((C, S, A, D, file))

with open("../data/raw.csv", "w") as f:
    writer = csv.writer(f)
    columns = ["C", "S", "A", "D", "frame_num", "img_path"]

    for point_id in POINT_IDS:
        columns.append(f"{point_id}_u")
        columns.append(f"{point_id}_v")

    if include_xyz:
        for point_id in POINT_IDS:
            columns.append(f"{point_id}_x")
            columns.append(f"{point_id}_y")
            columns.append(f"{point_id}_z")

    writer.writerow(columns)

    for C, S, A, D, file in tqdm.tqdm(FILES):
        frames = process_file(file)

        for frame in frames:
            frame_num, img_path, xy, xyz = frame
            columns = [C, S, A, D, frame_num, img_path]
            for i in range(len(POINT_IDS)):
                columns.append(xy[i][0])
                columns.append(xy[i][1])
            if include_xyz:
                for i in range(len(POINT_IDS)):
                    columns.append(xyz[i][0])
                    columns.append(xyz[i][1])
                    columns.append(xyz[i][2])
            writer.writerow(columns)
