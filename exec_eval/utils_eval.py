import torch
import numpy as np

from torch.utils.data import DataLoader

from class_model.tpcnn import TPCNN
from utils.display import print_header
from class_dataloader.dataloader import Train
from exec_train.utils_model import load_checkpoint

# Get spatial keypoint
def get_spatial_keypoint(keypoint):
    b = np.reshape(np.array([-100, -100, -1800]), (1,1,3))
    resolution = 100
    max = 19
    spatial_keypoint = keypoint * max * resolution + b
    return spatial_keypoint

# Get keypoint spatial distance
def get_keypoint_spatial_dis(keypoint_GT, keypoint_pred):
    dis = get_spatial_keypoint(keypoint_pred) - get_spatial_keypoint(keypoint_GT)
    return dis

# Model Pass
def model_generate(model, batch, configs):
    # Set data to device
    idx = torch.tensor(batch[0], dtype=torch.float, device=configs.device)
    tactile = torch.tensor(batch[1], dtype=torch.float, device=configs.device)
    heatmap = torch.tensor(batch[2], dtype=torch.float, device=configs.device)
    keypoint = torch.tensor(batch[3], dtype=torch.float, device=configs.device)

    # Generate
    heatmap_transform, keypoint_out = model.generate(tactile, heatmap, keypoint)

    # Return
    return heatmap_transform, keypoint_out, tactile, heatmap, keypoint

# Setup Model
def setup_model(configs):
    # Initialize start epoch
    start_epoch = 0

    # Initialize model
    print_header("Initialize Model")
    model = TPCNN(configs=configs)
    model = model.to(device=configs['device'])

    # Load checkpoint model
    model, checkpoint = load_checkpoint(model, configs['eval_checkpoint'])

    # Return
    return model

# Setup data
def setup_data(configs):
    # Initialize dataloader
    print_header("Initialize Dataloader")

    # Create dataset
    test_dataset = Train(configs=configs)

    # Create dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=False, drop_last=False)

    # Return
    return test_dataloader