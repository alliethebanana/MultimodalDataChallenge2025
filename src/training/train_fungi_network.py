"""

Training a model 


"""


import os
import random
import time
import csv

import torch
import torch.nn as nn
from torch.optim import Adam

from tqdm import tqdm

import numpy as np

from src.data import load_data
from src.config.model_config import ModelConfig
from src.model.complete_model import CompleteModel



def ensure_folder(folder):
    """
    Ensure a folder exists; if not, create it.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)


def seed_torch(seed=777):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize_csv_logger(file_path):
    """Initialize the CSV file with header."""
    header = ["epoch", "time", "val_loss", "val_accuracy", "train_loss", "train_accuracy"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)


def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy):
    """Log epoch summary to the CSV file."""
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, epoch_time, val_loss, val_accuracy, train_loss, train_accuracy])


def train_fungi_network(
        data_file: str, image_path: str, checkpoint_dir: str, 
        model_config: ModelConfig):
    """
    Train the network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Ensure seed set 
    seed_torch()

    # Set Logger
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)


    train_loader, valid_loader = load_data.get_train_dataloaders(
        data_file, image_path, model_config.image_embedding_type == 'dino')

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompleteModel(model_config)
    model.to(device)
    model.clip.to(device)

    # Define Optimization, Scheduler, and Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # Label smoothing for CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Early stopping setup
    patience = model_config.patience
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # Training Loop
    # pylint: disable=not-callable
    for epoch in tqdm(range(100)):  # Maximum epochs
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Training Loop
        for images, labels, _, md in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(images, md, device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate train accuracy
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)
        
        # Calculate overall train accuracy and average loss
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        
        # Validation Loop
        with torch.no_grad():
            for images, labels, _, md in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model.forward(images, md, device)
                val_loss += criterion(outputs, labels).item()
                
                # Calculate validation accuracy
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)

        # Calculate overall validation accuracy and average loss
        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)

        # Stop epoch timer and calculate elapsed time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Print summary at the end of the epoch
        print(f"Epoch {epoch + 1} Summary: "
            f"Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, "
            f"Epoch Time = {epoch_time:.2f} seconds")
        
        # Log epoch metrics to the CSV file
        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        # Save Models Based on Accuracy and Loss
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"Epoch {epoch + 1}: Best loss updated to {best_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break

def evaluate_network_on_test_set(
        data_file, image_path, checkpoint_dir, session_name, model_config: ModelConfig):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    test_loader = load_data.get_test_dataloader(
        data_file, image_path, model_config.image_embedding_type == 'dino')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompleteModel(model_config)
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames, md in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model.forward(images, md, device).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")


def evaluate_network_on_final_set(
        data_file, image_path, checkpoint_dir, session_name, model_config: ModelConfig):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "final_predictions.csv")

    test_loader = load_data.get_final_dataloader(
        data_file, image_path, model_config.image_embedding_type == 'dino')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompleteModel(model_config)
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames, md in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model.forward(images, md, device).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")
