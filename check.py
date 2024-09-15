from utils.visualise import visualize_output
from torch.utils.data import DataLoader, random_split, Subset
import yaml
import torch
from dataset import PoseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_model(config):
    if config["model"]["name"] == "HRNet":
        from models.HRNet.hrnet import PoseHighResolutionNet

        model = PoseHighResolutionNet(config["model"]["config"])
        if config["model"]["weights"]:
            model.init_weights(config["model"]["weights"])
    else:
        raise ValueError(
            f"Model {config['model']['name']} not recognized"
        )  # Added error handling
    return model


config = load_config("config.yaml")
dataset = PoseDataset(config["dataset"])
max_items = min(len(dataset), 100)
dataset = Subset(dataset, list(range(max_items)))
val_split = config["dataset"]["val_split"]
train_size = int((1 - val_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
# batch_size = config["training"]["batch_size"]
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = get_model(config).to(device)
# optimizer = get_optimizer(config, model)
# criterion = nn.MSELoss()

# Training loop
num_epochs = config["training"]["epochs"]


for batch_idx, (images, targets, keypoints) in enumerate(train_loader):
    images, targets = images.to(device), targets.to(device)

    outputs = model(images)

    image = images[0].permute(1, 2, 0).cpu().numpy()
    target = targets[0].cpu().numpy()
    output = outputs[0].detach().cpu().numpy()
    keypoint = keypoints[0].cpu().numpy()
    visualize_output(image, keypoint, target)
    visualize_output(image, keypoint, output)
