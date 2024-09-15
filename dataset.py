import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

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


class PoseDataset(Dataset):
    def __init__(self, config, file):
        self.img_dir = config["img_dir"]
        self.keypoints = config["keypoints"]
        self.image_size = config["preprocess"]["resize"]
        self.output_size = config["preprocess"]["output_size"]
        self.sigma = config["sigma"]

        self.file = file
        self.data = pd.read_csv(self.file)
        self.keypoint_columns = [
            col for col in self.data.columns if "_u" in col or "_v" in col
        ]

        self.keypoint_indices = []
        for kp in self.keypoints:
            self.keypoint_indices.append(all_keypoints.index(kp))

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        image, keypoints = self.load_sample(idx)

        image, keypoints, keypoint_visibility = self.preprocess(image, keypoints)

        heatmaps = self.generate_heatmaps(keypoints, keypoint_visibility)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        heatmaps = torch.from_numpy(heatmaps).float()

        return image, heatmaps, keypoints, keypoint_visibility

    def preprocess(self, image, keypoints):
        keypoints[:, 0] = keypoints[:, 0] * self.image_size[0] / image.shape[1]
        keypoints[:, 1] = keypoints[:, 1] * self.image_size[1] / image.shape[0]

        image = cv2.resize(image, tuple(self.image_size))
        image = image / 255.0

        keypoints[np.isnan(keypoints)] = 0

        keypoint_visibility = keypoints[:, 2] > 0
        keypoint_visibility = keypoint_visibility.astype(np.float32)

        keypoints = keypoints.astype(np.float32)
        keypoints = keypoints[:, :2]

        return image, keypoints, keypoint_visibility

    def generate_heatmaps(self, keypoints, keypoint_visibility):
        heatmaps = np.zeros(
            (len(keypoints), self.output_size[1], self.output_size[0]), dtype=np.float32
        )

        # normalize keypoints
        normalized_keypoints = np.copy(keypoints)
        normalized_keypoints[:, 0] = normalized_keypoints[:, 0] / self.image_size[0]
        normalized_keypoints[:, 1] = normalized_keypoints[:, 1] / self.image_size[1]

        for i, keypoint in enumerate(normalized_keypoints):
            if not keypoint_visibility[i]:
                continue
            heatmaps[i] = self.gaussian_2d(self.output_size, keypoint, self.sigma)

        return heatmaps

    def gaussian_2d(self, size, center, sigma):
        x = np.arange(0, size[0], 1, np.float32)
        y = np.arange(0, size[1], 1, np.float32)
        y = y[:, np.newaxis]
        x0, y0 = center

        x0 = x0 * size[0]
        y0 = y0 * size[1]

        # TODO: handle keypoints that are not visible v = 0

        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

        gaussian = gaussian / np.max(gaussian)

        gaussian[gaussian < 0] = 0
        gaussian[np.isnan(gaussian)] = 0

        return gaussian

    def load_sample(self, idx):
        row = self.data.iloc[idx]
        img_path = self.data.iloc[idx]["cropped_img_path"]
        img_path = os.path.join(self.img_dir, img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = self.get_keypoints(
            row[self.keypoint_columns], image.shape[1], image.shape[0]
        )
        keypoints = self.filter_keypoints(keypoints)

        return image, keypoints

    def get_keypoints(self, keypoints_str, img_width, img_height):
        keypoints = []
        for i in range(0, len(keypoints_str), 2):
            x = keypoints_str.iloc[i]
            y = keypoints_str.iloc[i + 1]
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

    def filter_keypoints(self, keypoints):
        return keypoints[self.keypoint_indices]
