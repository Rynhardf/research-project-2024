import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
import os
from dataset import PoseDataset
import cv2
from utils import visualize_output
import time
from datetime import datetime
import json
from utils import JointsMSELoss, load_config, save_config, load_model
from utils import get_keypoints_from_heatmaps, normalized_mae_in_pixels
import argparse


def get_optimizer(config, model):
    optimizer_name = config["training"]["optimizer"].lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=config["training"]["learning_rate"], momentum=0.9
        )
    else:
        raise ValueError(
            f"Optimizer {config['training']['optimizer']} not recognized"
        )  # Added error handling
    return optimizer


def save_checkpoint(checkpoint_dir, epoch, results, weights):
    checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

    results_path = os.path.join(checkpoint_dir, "results.json")
    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    weights_path = os.path.join(checkpoint_dir, f"weights_epoch_{epoch}.pth")
    torch.save(weights, weights_path)

    print(f"Checkpoint saved to {checkpoint_dir}")


def validate(model, val_loader, criterion, device, input_size):
    model.eval()
    val_loss = 0
    norm_mae = 0
    with torch.no_grad():
        for images, targets, gt_keypoints, keypoint_visibility in val_loader:
            images, targets = images.to(device), targets.to(device)
            gt_keypoints = gt_keypoints.to(device)
            keypoint_visibility = keypoint_visibility.to(device)
            outputs = model(images)
            pred_keypoints = get_keypoints_from_heatmaps(
                outputs.detach(), input_size[::-1]
            )
            loss = criterion(outputs, targets, keypoint_visibility)
            val_loss += loss.item()

            norm_mae += normalized_mae_in_pixels(
                pred_keypoints, gt_keypoints, input_size, keypoint_visibility
            )

    val_loss /= len(val_loader)
    norm_mae /= len(val_loader)
    model.train()  # Set model back to training mode
    return val_loss, norm_mae


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    train_dataset = PoseDataset(config["dataset"], config["dataset"]["train"])
    val_dataset = PoseDataset(config["dataset"], config["dataset"]["val"])
    max_items = min(len(train_dataset), config["dataset"]["max_items"])
    train_dataset = Subset(train_dataset, list(range(max_items)))

    # Data loaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(config["model"]).to(device)
    optimizer = get_optimizer(config, model)
    criterion = JointsMSELoss()

    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    output_dir = os.path.join(config["training"]["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save the configuration before starting training
    save_config(config, os.path.join(output_dir, "config.yaml"))

    # Training loop
    num_epochs = config["training"]["epochs"]
    best_val_loss = float("inf")

    # To log results
    results = {"epochs": []}

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batch_losses = []  # List to store batch-level losses
        batch_times = []  # List to store batch processing times

        epoch_start_time = time.time()  # Start time of the epoch

        for batch_idx, batch in enumerate(train_loader):
            (images, targets, gt_keypoints, keypoint_visibility) = batch
            batch_start_time = time.time()  # Start time of the batch

            images, targets = images.to(device), targets.to(device)
            gt_keypoints = gt_keypoints.to(device)
            keypoint_visibility = keypoint_visibility.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            pred_keypoints = get_keypoints_from_heatmaps(
                outputs.detach(), config["model"]["input_size"][::-1]
            )

            loss = criterion(outputs, targets, keypoint_visibility)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            batch_losses.append(loss.item())  # Save batch loss

            batch_time = time.time() - batch_start_time  # Time taken for the batch
            batch_times.append(batch_time)  # Save batch time

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                f"Loss: {loss.item():.6f}, Batch Time: {batch_time:.2f}s"
            )

        avg_train_loss = total_train_loss / len(train_loader)

        epoch_time = time.time() - epoch_start_time  # Total time for the epoch

        val_loss, norm_mae = validate(
            model, val_loader, criterion, device, config["model"]["input_size"]
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.6f}, "
            f"Validation Loss: {val_loss:.6f}, Normalized MAE: {norm_mae:.6f}, Epoch Time: {epoch_time:.2f}s"
        )

        # Save epoch results
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "normalized_mae": norm_mae,
            "batch_losses": batch_losses,
            "batch_times": batch_times,
            "epoch_time": epoch_time,
        }

        results["epochs"].append(epoch_result)

        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving model..."
            )
            best_val_loss = val_loss
            save_checkpoint(
                output_dir,  # Use the timestamped output directory
                epoch + 1,
                {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                },
                model.state_dict(),
            )

    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as file:
        json.dump(
            results, file, indent=4
        )  # 'indent=4' makes the JSON file more readable
    print(f"Training complete. Results saved to {results_path}")
