"""
This script implements a PyTorch deep learning training pipeline for an eye tracking application.
It includes a main function to pass in arguments, train and validation functions, and uses MLflow as the logging library.
The script also supports fine-grained deep learning hyperparameter tuning using argparse and JSON configuration files.
All hyperparameters are logged with MLflow.

Contributors: Zuowen Wang, Chang Gao
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich, TU Delft
Email: wangzu@ethz.ch, gaochangw@outlook.com
"""

import argparse, json, os, mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from torch.utils.data import DataLoader
from model.BaselineEyeTrackingModel import CNN_GRU
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
    LabelTemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# Add this function to clean cached dataset
def clean_cached_dataset():
    cache_dir = './cached_dataset'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleaned cached dataset directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(args.num_epochs):
        # Add progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}', leave=True)
        model, train_loss, metrics = train_epoch(model, train_pbar, criterion, optimizer, args)
        
        # Create metrics table with clear training/validation status
        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            table = Table(title=f"Epoch {epoch+1}/{args.num_epochs} - Train & Val Metrics")
        else:
            table = Table(title=f"Epoch {epoch+1}/{args.num_epochs} - Train Metrics")
            
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Add training metrics
        table.add_row("Train Loss", f"{train_loss:.4f}")
        for metric_name, value in metrics['tr_p_euc_error_all'].items():
            table.add_row(f"Train {metric_name}", f"{value:.4f}")
        for metric_name, value in metrics['tr_p_acc_all'].items():
            table.add_row(f"Train {metric_name}", f"{value:.4f}")

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metrics(metrics['tr_p_acc_all'], step=epoch)
        mlflow.log_metrics(metrics['tr_p_euc_error_all'], step=epoch)

        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            # Add progress bar for validation
            val_pbar = tqdm(val_loader, desc=f'Validation', leave=True)
            val_loss, val_metrics = validate_epoch(model, val_pbar, criterion, args)
            
            # Add validation metrics to table
            table.add_row("Validation Loss", f"{val_loss:.4f}")
            for metric_name, value in val_metrics['val_p_euc_error_all'].items():
                table.add_row(f"Val {metric_name}", f"{value:.4f}")
            for metric_name, value in val_metrics['val_p_acc_all'].items():
                table.add_row(f"Val {metric_name}", f"{value:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(mlflow.get_artifact_uri(), \
                            f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth")
                torch.save(model.state_dict(), best_model_path)
                rprint(f"[green]Saved best model to:[/green] {best_model_path}")
                
                # DANGER Zone, this will delete files (checkpoints) in MLflow artifact
                top_k_checkpoints(args, mlflow.get_artifact_uri())

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metrics(val_metrics['val_p_acc_all'], step=epoch)
            mlflow.log_metrics(val_metrics['val_p_euc_error_all'], step=epoch)

        # Print the metrics table
        console.print(table)

    return model


def main(args):
    # Clean cached dataset before starting
    clean_cached_dataset()
    
    # Load hyperparameters from JSON configuration file
    if args.config_file:
        with open(os.path.join('./configs', args.config_file), 'r') as f:
            config = json.load(f)
        # Overwrite hyperparameters with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    else:
        raise ValueError("Please provide a JSON configuration file.")

    # Set up MLflow logging
    mlflow.set_tracking_uri(args.mlflow_path)
    mlflow.set_experiment(experiment_name=args.experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # dump this training file to MLflow artifact
        mlflow.log_artifact(__file__)

        # Log all hyperparameters to MLflow
        mlflow.log_params(vars(args))
        # also dump the args to a JSON file in MLflow artifact
        with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
            json.dump(vars(args), f)

        # Define your model, optimizer, and criterion
        model = eval(args.architecture)(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.loss == "mse":
            criterion = nn.MSELoss()
        elif args.loss == "weighted_mse":
            criterion = weighted_MSELoss(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device), \
                                         reduction='mean')
        else:
            raise ValueError("Invalid loss name")

        factor = args.spatial_factor # spatial downsample factor
        temp_subsample_factor = args.temporal_subsample_factor

        # First we define the label transformations
        label_transform = transforms.Compose([
            ScaleLabel(factor),
            NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
        ])

        # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
        train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", \
                        transform=transforms.Downsample(spatial_factor=factor),
                        target_transform=label_transform
                                                  )
        val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                        transform=transforms.Downsample(spatial_factor=factor),
                        target_transform=label_transform
                                                )

        # Then we slice the event recordings into sub-sequences. 
        # The time-window is determined by the sequence length (train_length, val_length) 
        # and the temporal subsample factor.
        slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
        train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds

        train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                        seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=False)
        # the validation set is sliced to non-overlapping sequences
        val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=0, \
                        seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=False)

        # After slicing the raw event recordings into sub-sequences, 
        # we make each subsequences into your favorite event representation, 
        # in this case event voxel-grid
        post_slicer_transform = transforms.Compose([
            SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
            EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), \
                                    n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
        ])

        # We use the Tonic SlicedDataset class to handle the collation of the sub-sequences into batches.
        train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
        val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}")

        # cache the dataset to disk to speed up training. The first epoch will be slow, but the following epochs will be fast.
        train_data = DiskCachedDataset(train_data, cache_path=f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}')
        val_data = DiskCachedDataset(val_data, cache_path=f'./cached_dataset/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}')

        # Finally we wrap the dataset with pytorch dataloader
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=int(os.cpu_count()-2), pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, \
                                num_workers=int(os.cpu_count()-2))

        # Train your model
        model = train(model, train_loader, val_loader, criterion, optimizer, args)

        # Save your model for the last epoch
        torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), f"model_last_epoch{args.num_epochs}.pth"))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training management arguments     
    parser.add_argument("--mlflow_path", type=str, help="path to MLflow tracking server")
    parser.add_argument("--experiment_name", type=str, help="name of the experiment")
    parser.add_argument("--run_name", type=str, help="name of the run")
    
    # a config file 
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")

    # training hyperparameters
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_epochs", type=int, help="number of epochs")
    
    args = parser.parse_args()

    main(args)
