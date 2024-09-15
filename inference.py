import yaml
import torch
import argparse
import cv2
import numpy as np


# get config file path as argument
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("image", type=str)
args = parser.parse_args()

config = load_config(args.config)
model = load_model(config["model"])

input_size = config["model"]["input_size"]
output_size = config["model"]["output_size"]

# Load image
image = cv2.imread(args.image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, input_size)

# Preprocess image
image = image / 255.0
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

# Inference
model.eval()
with torch.no_grad():
    output = model(image)
    output = output.squeeze(0).numpy()

# Display output
org_image = cv2.imread(args.image)
org_size = (org_image.shape[1], org_image.shape[0])

heatmap = None
for i in range(output.shape[0]):
    hm = cv2.resize(output[i], org_size)

    # add all heatmaps together
    if heatmap is None:
        heatmap = hm
    else:
        heatmap += hm

# Normalize heatmap
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

# Overlay heatmap on image
result = cv2.addWeighted(org_image, 0.5, heatmap, 0.5, 0)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
