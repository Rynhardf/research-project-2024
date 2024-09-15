import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
import os
from dataset import PoseDataset
import cv2
from utils.visualise import visualize_output
import time
from datetime import datetime
import json
from utils.loss import JointsMSELoss
import argparse
from utils.utils import save_config, get_model


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

    weights_path = os.path.join(checkpoint_dir, "weights.pth")
    torch.save(weights, weights_path)

    print(f"Checkpoint saved to {checkpoint_dir}")


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets, keypoints in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    model.train()  # Set model back to training mode
    return val_loss


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    dataset = PoseDataset(config["dataset"])
    max_items = min(len(dataset), 100)
    dataset = Subset(dataset, list(range(max_items)))
    val_split = config["dataset"]["val_split"]
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(config["model"]).to(device)
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

        for batch_idx, (images, targets, keypoints) in enumerate(train_loader):
            batch_start_time = time.time()  # Start time of the batch

            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # image = images[0].permute(1, 2, 0).cpu().numpy()
            # target = targets[0].cpu().numpy()
            # output = outputs[0].detach().cpu().numpy()
            # keypoint = keypoints[0].cpu().numpy()
            # visualize_output(image, keypoint, target)
            # visualize_output(image, keypoint, output)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            batch_losses.append(loss.item())  # Save batch loss

            batch_time = time.time() - batch_start_time  # Time taken for the batch
            batch_times.append(batch_time)  # Save batch time

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, Batch Time: {batch_time:.2f}s"
            )

        avg_train_loss = total_train_loss / len(train_loader)

        epoch_time = time.time() - epoch_start_time  # Total time for the epoch

        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, Epoch Time: {epoch_time:.2f}s"
        )

        # Save epoch results
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "batch_losses": batch_losses,
            "batch_times": batch_times,
            "epoch_time": epoch_time,
        }

        results["epochs"].append(epoch_result)

        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model..."
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
