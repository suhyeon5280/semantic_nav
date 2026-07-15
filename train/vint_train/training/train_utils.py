import sys
# appending a path
path_mapcache = "/nfs/kun2/users/noriaki/map_cache/"
#path_mapcache = "/media/noriaki/Noriaki_Data2/map_cache"
sys.path.append(path_mapcache)
try:
    from map_cache import MapTileCache
except Exception:
    MapTileCache = None  # only needed for the satellite-map (frodobot) training paths

try:
    import wandb
except Exception:
    wandb = None  # optional
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import clip
import pickle
import cv2
import random
from PIL import Image
#from torch.cuda.amp import autocast

import psutil
import copy
import utm

def latlon_to_utm(lat, lon):
    """ Convert latitude and longitude to UTM coordinates. """
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return easting, northing, zone_number, zone_letter

def utm_to_latlon(easting, northing, zone_number, zone_letter):
    """ Convert UTM coordinates back to latitude and longitude. """
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    return lat, lon

def transform_position(lat, lon, heading, X, Y, theta):
    """
    Compute new latitude, longitude, and heading after moving by (X, Y, theta)
    in the local coordinate system where:
    - X is forward (aligned with heading)
    - Y is left (perpendicular to heading)
    - theta is counterclockwise (CCW)
    """
    # Convert lat/lon to UTM
    easting, northing, zone_number, zone_letter = latlon_to_utm(lat, lon)

    # Convert heading from degrees to radians
    # heading_rad = np.radians(heading)
    heading_rad = heading
    new_heading = (heading - theta) 
    
    # Corrected transformation: X moves forward, Y moves left
    delta_easting = np.sqrt(X**2 + Y**2) * np.sin(new_heading) 
    delta_northing = np.sqrt(X**2 + Y**2) * np.cos(new_heading) 

    # New position in UTM coordinates
    new_easting = easting + delta_easting
    new_northing = northing + delta_northing

    # Convert back to latitude and longitude
    new_lat, new_lon = utm_to_latlon(new_easting, new_northing, zone_number, zone_letter)

    # Update heading (subtract for CCW rotation)
    new_heading = (heading - theta)  

    return new_lat, new_lon, new_heading
        
# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def get_current_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]
    
# Train utils for ViNT and GNM
def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    return results

def _compute_losses_lan(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    pose_obj_label: torch.Tensor,
    pose_obj_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    image_solo: bool,
    sate_solo: bool,    
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    obj_loss = F.mse_loss(pose_obj_label, pose_obj_pred)
    act_smooth = F.mse_loss(action_pred[:,0:-1], action_pred[:,1:])
    
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "obj_loss": obj_loss,
        "smooth_loss": act_smooth,                
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    if image_solo:
        total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss + 0.05*act_smooth
    elif sate_solo:
        total_loss = (1 - alpha) * action_loss + 0.05*act_smooth
    else:
        total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss + 0.05*obj_loss + 0.05*act_smooth
    results["total_loss"] = total_loss

    return results

def _compute_losses_gps(
    action_label: torch.Tensor,
    action_pred: torch.Tensor,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = F.mse_loss(action_pred, action_label, reduction="mean")


    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = action_loss
    results["total_loss"] = total_loss

    return results

def geometry_criterion_weight(pc, rsize, step_size, weight_geo, device):
    #Input:
    #    pc: estimated point cloud on the virtual robto coordinate,  batch size x step_size x 3 x 128 x 416
    #    rsize: randomized robot radius, batch size x 1
    #    step_size: the number of the virtual control step (control horizon)
    #    device: device id (CPU or GPU)
    #
    #Output:
    #    average of the geometric loss
    
    pred_clamp = []
    pred_ref = []
    MSE = 0
    bias = 10
    bs, seq, nch, hs, ws = pc.size()
    #print(pc.size())

    pred = torch.sqrt(pc[:,:,0]**2 + pc[:,:,2]**2) 
    yaxis = pc[:,:,1]
    
    pc_1 = torch.cat((pc[:,:,:,0:1,:],pc[:,:,:,0:127,:]), axis=3)
    pc_2 = torch.cat((pc[:,:,:,1:128,:],pc[:,:,:,127:128,:]), axis=3)
    pc_3 = torch.cat((pc[:,:,:,:,0:1],pc[:,:,:,:,0:415]), axis=4)
    pc_4 = torch.cat((pc[:,:,:,:,1:416],pc[:,:,:,:,415:416]), axis=4)
    weight = (torch.sqrt(torch.sum(torch.square(pc_1 - pc_2), 2))) * (torch.sqrt(torch.sum(torch.square(pc_3 - pc_4), 2)))
    weight = (weight)[:,:,:,bias:416-bias]

    count = 0
    for i in range(bs):
        mask1 = (yaxis[i:i+1,:,:] < 0.15*torch.ones((1, step_size, 128, 416), device=device))
        mask2 = (yaxis[i:i+1,:,:] > -0.3*torch.ones((1, step_size, 128, 416), device=device))
            
        mask = torch.logical_and(mask1, mask2)[:,:,:,bias:416-bias]

        pred_cap = torch.clamp(pred[i:i+1,:,:], 0.0, rsize[i, 0].item())[:,:,:,bias:416-bias]
        pred_cap_mask = pred_cap[mask]
        weight_mask = weight[i:i+1][mask]
        weight_mask = torch.clamp(weight_mask, 0.0, 0.01)

        num_masked = torch.sum((pred_cap_mask == rsize[i, 0].item()).float())
            
        num_m = list(pred_cap_mask.size())[0]
        count = num_m - num_masked.cpu().float().item()
        pred_ref = rsize[i, 0].item()*torch.ones(num_m).to(device)

        MSE += weight_geo[i]*torch.sum(weight_mask*(pred_cap_mask - pred_ref)**2)/(count + 1e-7)*2.0e+3

    return MSE/bs

def geometry_criterion(pc, rsize, step_size, device):
    #Input:
    #    pc: estimated point cloud on the virtual robto coordinate,  batch size x step_size x 3 x 128 x 416
    #    rsize: randomized robot radius, batch size x 1
    #    step_size: the number of the virtual control step (control horizon)
    #    device: device id (CPU or GPU)
    #
    #Output:
    #    average of the geometric loss
    
    pred_clamp = []
    pred_ref = []
    MSE = 0
    bias = 10
    bs, seq, nch, hs, ws = pc.size()

    pred = torch.sqrt(pc[:,:,0]**2 + pc[:,:,2]**2) 
    yaxis = pc[:,:,1]
    
    pc_1 = torch.cat((pc[:,:,:,0:1,:],pc[:,:,:,0:127,:]), axis=3)
    pc_2 = torch.cat((pc[:,:,:,1:128,:],pc[:,:,:,127:128,:]), axis=3)
    pc_3 = torch.cat((pc[:,:,:,:,0:1],pc[:,:,:,:,0:415]), axis=4)
    pc_4 = torch.cat((pc[:,:,:,:,1:416],pc[:,:,:,:,415:416]), axis=4)
    weight = (torch.sqrt(torch.sum(torch.square(pc_1 - pc_2), 2))) * (torch.sqrt(torch.sum(torch.square(pc_3 - pc_4), 2)))
    weight = (weight)[:,:,:,bias:416-bias]

    count = 0
    for i in range(bs):
        mask1 = (yaxis[i:i+1,:,:] < 0.0*torch.ones((1, step_size, 128, 416), device=device))
        mask2 = (yaxis[i:i+1,:,:] > -0.3*torch.ones((1, step_size, 128, 416), device=device))
            
        mask = torch.logical_and(mask1, mask2)[:,:,:,bias:416-bias]

        pred_cap = torch.clamp(pred[i:i+1,:,:], 0.0, rsize[i, 0].item())[:,:,:,bias:416-bias]
        pred_cap_mask = pred_cap[mask]
        weight_mask = weight[i:i+1][mask]
        weight_mask = torch.clamp(weight_mask, 0.0, 0.01)

        num_masked = torch.sum((pred_cap_mask == rsize[i, 0].item()).float())
            
        num_m = list(pred_cap_mask.size())[0]
        count = num_m - num_masked.cpu().float().item()
        pred_ref = rsize[i, 0].item()*torch.ones(num_m).to(device)

        MSE += torch.sum(weight_mask*(pred_cap_mask - pred_ref)**2)/(count + 1e-7)*2.0e+3

    return MSE/bs

def geometry_criterion_range(pc, rsize, step_size, limit_range, device):
    #Input:
    #    pc: estimated point cloud on the virtual robto coordinate,  batch size x step_size x 3 x 128 x 416
    #    rsize: randomized robot radius, batch size x 1
    #    step_size: the number of the virtual control step (control horizon)
    #    device: device id (CPU or GPU)
    #
    #Output:
    #    average of the geometric loss
    
    pred_clamp = []
    pred_ref = []
    MSE = 0
    #bias = 10
    bias = 20
    bs, seq, nch, hs, ws = pc.size()
    #print(pc.size())

    pred = torch.sqrt(pc[:,:,0]**2 + pc[:,:,2]**2) 
    yaxis = pc[:,:,1]
    
    pc_1 = torch.cat((pc[:,:,:,0:1,:],pc[:,:,:,0:127,:]), axis=3)
    pc_2 = torch.cat((pc[:,:,:,1:128,:],pc[:,:,:,127:128,:]), axis=3)
    pc_3 = torch.cat((pc[:,:,:,:,0:1],pc[:,:,:,:,0:415]), axis=4)
    pc_4 = torch.cat((pc[:,:,:,:,1:416],pc[:,:,:,:,415:416]), axis=4)
    weight = (torch.sqrt(torch.sum(torch.square(pc_1 - pc_2), 2))) * (torch.sqrt(torch.sum(torch.square(pc_3 - pc_4), 2)))
    weight = (weight)[:,:,:,bias:416-bias]

    count = 0
    for i in range(bs):
        mask1 = (yaxis[i:i+1,:,:] < limit_range[i,0]*torch.ones((1, step_size, 128, 416), device=device))
        mask2 = (yaxis[i:i+1,:,:] > limit_range[i,1]*torch.ones((1, step_size, 128, 416), device=device))
            
        mask = torch.logical_and(mask1, mask2)[:,:,:,bias:416-bias]

        pred_cap = torch.clamp(pred[i:i+1,:,:], 0.0, rsize[i, 0].item())[:,:,:,bias:416-bias]
        pred_cap_mask = pred_cap[mask]
        weight_mask = weight[i:i+1][mask]
        weight_mask = torch.clamp(weight_mask, 0.0, 0.01)

        num_masked = torch.sum((pred_cap_mask == rsize[i, 0].item()).float())
            
        num_m = list(pred_cap_mask.size())[0]
        count = num_m - num_masked.cpu().float().item()
        pred_ref = rsize[i, 0].item()*torch.ones(num_m).to(device)

        MSE += torch.sum(weight_mask*(pred_cap_mask - pred_ref)**2)/(count + 1e-7)*2.0e+3

    return MSE/bs

def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        
        goal_image = transform(goal_image).to(device)
        model_outputs = model(obs_image, goal_image)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()
      
        dist_pred, action_pred = model_outputs

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,

):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            model_outputs = model(obs_image, goal_image)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # Log data to wandb/console, with visualizations selected from the last batch
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results
    
def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)
        
def twist_to_pose_diff_torch(v, w, dt):
    """integrate 2D twist to get pose difference.

    Assuming constant velocity during time period `dt`.

    Args:
        v (float): velocity
        w (float): angular velocity
        dt (float): time delta

    """

    theta = -w  * dt
    z = v * dt * sinc_apx(-theta / np.pi)
    x = -v * dt * sinc_apx(-theta / (2 * np.pi)) * torch.sin(-theta / 2)
    return x, z, theta

def robot_pos_model_fix(linear_vel, angular_vel):
    # velocity commands integral
    bs, chorizon = linear_vel.shape
    device = linear_vel.device

    px = []
    pz = []
    pyaw = []
    Tacc = torch.eye(4, 4).unsqueeze(0).repeat(bs,1,1).to(device)
    for i in range(chorizon):
        x, z, yaw = twist_to_pose_diff_torch(linear_vel[:, i], angular_vel[:, i], 0.333)
        Todom = torch.zeros((bs, 4, 4)).to(device)
        Todom[:, 0, 0] = torch.cos(yaw)
        Todom[:, 0, 2] = torch.sin(yaw)
        Todom[:, 1, 1] = 1.0
        Todom[:, 2, 0] = -torch.sin(yaw)
        Todom[:, 2, 2] = torch.cos(yaw)
        Todom[:, 0, 3] = x
        Todom[:, 2, 3] = z
        Todom[:, 3, 3] = 1.0        
        
        Tacc = torch.matmul(Tacc, Todom)
               
        pyaw.append(torch.arctan(Tacc[:, 0, 2]/(Tacc[:, 0, 0] + 0.000000001)))        
        px.append(Tacc[:, 0, 3])
        pz.append(Tacc[:, 2, 3])        
    return px, pz, pyaw    

def train_lelan(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        wandb_log_freq: how often to log with wandb
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    model.eval_text_encoder()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", "train", window_size=print_log_freq)    
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                obj_poses,
                obj_inst,
                goal_pos_norm,                
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
                            
            batch_obs_images = transform(obs_image).to(device)
            batch_obj_poses = obj_poses.to(device)
            
            batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
            
            with torch.no_grad():  
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
            
            B = batch_obs_images.shape[0]
            
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32))
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
 
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_obj_poses)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
            
            # Total loss
            loss = dist_loss + 1.0*diff_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total loss": loss_cpu})
            wandb.log({"pose loss": dist_loss.item()})
            wandb.log({"vel smooth loss": diff_loss.item()})

            if i % print_log_freq == 0:
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()                 
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    obj_poses,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )

def train_lelan_col(
    model: nn.Module,
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,    
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    weight_col_loss: float,    
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        ema_model_nomad: exponential moving average model of pre-trained NoMaD policy for cropped goal image        
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        weight_col_loss: weight for collision avoindace loss
        epoch: current epoch
        print_log_freq: how often to print loss
        wandb_log_freq: how often to log with wandb
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    model.eval_text_encoder()
    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()    
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", "train", window_size=print_log_freq)    
    col_loss_logger = Logger("col loss", "train", window_size=print_log_freq)       
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
        "col loss": col_loss_logger,        
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                goal_pos,
                obj_inst,
                goal_pos_norm,                
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
                                                      
            batch_obs_current = transform(obs_image).to(device)

            batch_goal_pos = goal_pos.to(device)
            batch_goal_pos_norm = goal_pos_norm.to(device)      
                        
            batch_obs_images = [transform(TF.resize(obs, (96, 96), antialias=True)) for obs in obs_images_list]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(TF.resize(goal_image, (96, 96), antialias=True)).to(device)
            
            batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
            
            B = batch_obs_images.shape[0]
            action_mask = torch.ones(B).to(device)
                        
            # split into batches
            batch_obs_images_list = torch.split(batch_obs_images, B, dim=0)
            batch_goal_images_list = torch.split(batch_goal_images, B, dim=0)

            with torch.no_grad():
                select_traj = supervision_from_nomad(
                    ema_model_nomad,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    batch_goal_pos_norm,
                    device,
                    project_folder,
                    epoch,
                    B,
                    i,                
                    30,
                    use_wandb,
                    )    
            
            with torch.no_grad():
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
                                                
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32), current_img=batch_obs_current)
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)

            #transformation from camera coordinate to robot coordinate
            px_ref_listx = []
            pz_ref_listx = []
            for it in range(8):
                px_ref_listx.append(px_ref_list[it].unsqueeze(1).unsqueeze(2))
                pz_ref_listx.append(pz_ref_list[it].unsqueeze(1).unsqueeze(2))
            traj_policy = torch.concat((torch.concat(pz_ref_listx, axis=1), -torch.concat(px_ref_listx, axis=1)), axis=2)
                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
            
            mask_nomad = (batch_goal_pos[:,1:2] > 1.0).float().unsqueeze(1).repeat(1,8,2)
            mask_dist = (~(batch_goal_pos[:,1:2] > 1.0)).float()
            sum_dist = mask_dist.sum()            
            col_loss = nn.functional.mse_loss(mask_nomad*traj_policy, 0.12*mask_nomad*select_traj)*float(B)/(float(B) - sum_dist.float() + 1e-7) #0.12 is de-normalization
            
            loss = 1.0*dist_loss + 1.0*diff_loss + weight_col_loss*col_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total loss": loss_cpu})
            wandb.log({"pose loss": dist_loss.item()})
            wandb.log({"vel smooth loss": diff_loss.item()})
            wandb.log({"col loss": col_loss.item()})
            
            if i % print_log_freq == 0:
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()                 
                losses['col loss'] = col_loss.item()       
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_col_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    (0.12*select_traj).cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )

def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
            ) = data

            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            # Get distance label
            distance = distance.float().to(device)

            deltas = get_delta(actions)         
            ndeltas = normalize_data(deltas, ACTION_STATS)         
            naction = from_numpy(ndeltas).to(device)                 
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)         
                        
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss # mse between ground truth noise and predicted noise

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})


            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )

###
def train_exaug_dist_gnm_delay(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    model_depth,
    #model_pedtraj,
    device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    distall_loss_logger = Logger("distall_loss", "train", window_size=print_log_freq)    
    smooth_loss_logger = Logger("smooth_loss", "train", window_size=print_log_freq)
    geo_loss_logger = Logger("geo_loss", "train", window_size=print_log_freq)
    social_loss_logger = Logger("social_loss", "train", window_size=print_log_freq)
    personal_loss_logger = Logger("personal_loss", "train", window_size=print_log_freq)
    disttemp_loss_logger = Logger("disttemp_loss", "train", window_size=print_log_freq)
        
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "distall_loss": distall_loss_logger,        
        "smooth_loss": smooth_loss_logger,
        "geo_loss": geo_loss_logger,
        "social_loss": social_loss_logger,
        "personal_loss": personal_loss_logger,        
        "disttemp_loss": disttemp_loss_logger,          
    }
    
    mask_360 = np.loadtxt(open("./mask_360view.csv", "rb"), delimiter=",", skiprows=0)           
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    dataloader_sub_iter = iter(dataloader_sub)

    linear_vel_old = 0.5*torch.rand(300, 8).float().to(device)
    angular_vel_old = 1.0*torch.rand(300, 8).float().to(device)
                  
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                _,
                _,
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                    current_image_depth_sub,
                    geoloss_range_sub,
                    local_goal_mat_sub,
                    local_yaw_sub,
                ) = next(dataloader_sub_iter)                
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                    current_image_depth_sub,
                    geoloss_range_sub,
                    local_goal_mat_sub,
                    local_yaw_sub,
                ) = next(dataloader_sub_iter)   
                
            Bf, _, _, _ = goal_image.size()
            Bg, _, _, _ = goal_image_sub.size()

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE)
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)
            
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
            current_image_depth_sub = current_image_depth_sub.to(device2)
            goal_image_sub = transform(goal_image_sub).to(device)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            local_goal_mat_sub = local_goal_mat_sub.to(device)
            local_yaw_sub = local_yaw_sub.to(device)

            ang_yaw_sub = []
            for iy in range(Bg):
                if local_yaw_sub[iy] % (2*3.14) > 3.14:
                    ang_yaw_sub.append(local_yaw_sub[iy] % (2*3.14) - 2.0*3.14)
                else:
                    ang_yaw_sub.append(local_yaw_sub[iy] % (2*3.14))            
            ang_yaw_sub_tensor = torch.tensor(ang_yaw_sub).to(device)
                                    
            distance_sub = dist_label_sub.float().to(device)
            fargoal_mask_sub = ((torch.abs(local_goal_mat_sub[:, 0,2]) < 2.0) * (torch.abs(local_goal_mat_sub[:, 1,2]) < 2.0)).to(device) * (torch.abs(ang_yaw_sub_tensor) < 2.0)                        
            goal_mask_sub = (dist_label_sub > 0.1) * fargoal_mask_sub
                                    
            local_goal_mat_sub[:, 0,2] *= 2.0                    
            local_goal_mat_sub[:, 1,2] *= 2.0  
            local_goal_vec_sub = local_goal_mat_sub.unsqueeze(1).repeat(1,8,1,1) 
                                                
            current_image_depth = (current_image.to(device2))*mask_360_torch
            current_image_depth = current_image_depth.to(device2)
            
            obs_images = torch.split(obs_image, 3, dim=1) 
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])            
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
                        
            combined_current_image_depth = torch.cat((current_image_depth, current_image_depth_sub), axis=0)        
            with torch.no_grad():
                #depth estimation
                proj_3d, outputs = model_depth.forward(combined_current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device)   
            

            ang_yaw = []
            for iy in range(Bf):
                if local_yaw[iy] % (2*3.14) > 3.14:
                    ang_yaw.append(local_yaw[iy] % (2*3.14) - 2.0*3.14)
                else:
                    ang_yaw.append(local_yaw[iy] % (2*3.14))
            
            ang_yaw_tensor = torch.tensor(ang_yaw).to(device)

            # Get distance label
            distance_metric = torch.sqrt(goal_pos.to(device)[:,0]**2 + goal_pos.to(device)[:,1]**2)
            fargoal_mask = ((torch.abs(local_goal_mat[:, 0,2]) < 5.0) * (torch.abs(local_goal_mat[:, 1,2]) < 5.0)).to(device)
            
            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * fargoal_mask
            goal_mask_zero = distance > 0.1

            for ig in range(Bf):
                if not goal_mask_zero[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, Bf-1) 
                    while ig == igr:
                        igr = random.randint(0, Bf-1) 
                    goal_image[ig] = goal_image[igr]
                    #print(ig, igr)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)

            rsize = torch.rand(Bf+Bg, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.randint(0, 5, (Bf+Bg, 1, 1)).to(device)     

            cs = random.randint(0,2)
            linear_vel_old_p = linear_vel_old[:, cs:cs+6]
            angular_vel_old_p = angular_vel_old[:, cs:cs+6]

            vel_past = torch.cat((linear_vel_old_p, angular_vel_old_p), axis=1).unsqueeze(2)                                  
            linear_vel, angular_vel, dist_temp = model(combined_obs_image, combined_goal_image, rsize, delay, vel_past)                           

            for ig in range(Bf+Bg):
                linear_vel_old_p[ig, delay[ig,0,0]:6] *= 0.0
                angular_vel_old_p[ig, delay[ig,0,0]:6] *= 0.0
                                
            linear_vel_d = torch.cat((linear_vel_old_p, linear_vel), axis=1)
            angular_vel_d = torch.cat((angular_vel_old_p, angular_vel), axis=1)            
            linear_vel_old = linear_vel.detach()
            angular_vel_old = angular_vel.detach()
               
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((pz_ref.unsqueeze(1), -px_ref.unsqueeze(1)), axis=1) #from camera coordinate to robot local coordinate
            
            mat_1 = torch.cat((torch.cos(-ry_ref).unsqueeze(1), -torch.sin(-ry_ref).unsqueeze(1), 2.0 * pz_ref.unsqueeze(1)), axis=1)
            mat_2 = torch.cat((torch.sin(-ry_ref).unsqueeze(1), torch.cos(-ry_ref).unsqueeze(1), -2.0 * px_ref.unsqueeze(1)), axis=1)
            mat_3 = torch.cat((torch.zeros(Bf+Bg,1), torch.zeros(Bf+Bg,1), torch.ones(Bf+Bg,1)), axis=1).to(device)   
            last_pose_mat = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
            
            robot_traj_list = []
            for ip in range(len(px_ref_list)):
                mat_1 = torch.cat((torch.cos(-ry_ref_list[ip]).unsqueeze(1), -torch.sin(-ry_ref_list[ip]).unsqueeze(1), 2.0 * pz_ref_list[ip].unsqueeze(1)), axis=1)
                mat_2 = torch.cat((torch.sin(-ry_ref_list[ip]).unsqueeze(1), torch.cos(-ry_ref_list[ip]).unsqueeze(1), -2.0 * px_ref_list[ip].unsqueeze(1)), axis=1)
                mat_3 = torch.cat((torch.zeros(Bf+Bg,1), torch.zeros(Bf+Bg,1), torch.ones(Bf+Bg,1)), axis=1).to(device)   
                mat_combine = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
                robot_traj_list.append(mat_combine.unsqueeze(1))
            robot_traj_vec = torch.cat(robot_traj_list, axis=1)
                                 
            local_goal_mat[:, 0,2] *= 2.0                    
            local_goal_mat[:, 1,2] *= 2.0  
            local_goal_vec = local_goal_mat.unsqueeze(1).repeat(1,8,1,1) 
                       
            combined_local_goal_mat = torch.cat((local_goal_mat.to(device), local_goal_mat_sub), axis=0)
            combined_local_goal_vec = torch.cat((local_goal_vec.to(device), local_goal_vec_sub), axis=0)
            combined_distance = torch.cat((distance, distance_sub), axis=0)
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)
            
            geoloss_range = torch.cat((0.0*torch.ones(Bf,1), -0.3*torch.ones(Bf,1)), axis=1)
            combined_geoloss_range = torch.cat((geoloss_range, geoloss_range_sub), axis=0).to(device)
             
            dist_loss = nn.functional.mse_loss(last_pose_mat[combined_goal_mask], combined_local_goal_mat.to(device)[combined_goal_mask])      
            dist_loss_f = nn.functional.mse_loss(last_pose_mat[0:50][combined_goal_mask[0:50]], combined_local_goal_mat.to(device)[0:50][combined_goal_mask[0:50]])    
            dist_loss_g = nn.functional.mse_loss(last_pose_mat[50:100][combined_goal_mask[50:100]], combined_local_goal_mat.to(device)[50:100][combined_goal_mask[50:100]])    
       
            distall_loss = nn.functional.mse_loss(robot_traj_vec[combined_goal_mask, 6:14], combined_local_goal_vec.to(device)[combined_goal_mask])                                    
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1][combined_goal_mask], linear_vel[:,1:][combined_goal_mask]) + nn.functional.mse_loss(angular_vel[:,:-1][combined_goal_mask], angular_vel[:,1:][combined_goal_mask]) 

            # Predict distance          
            dist_temp_loss = F.mse_loss(dist_temp.squeeze(-1), combined_distance)
            
            norm = 10.0
            """
            if sacson:
                ped_past = torch.cat((-torch.flip(ped_list[:,:,1], dims=[1]), torch.flip(ped_list[:,:,0], dims=[1])), axis=1).to(device)  
                robot_past = torch.cat((-torch.flip(robot_list[:,:,1], dims=[1]), torch.flip(robot_list[:,:,0], dims=[1])), axis=1).to(device)
                
                robot_future_x = []
                robot_future_z = []
                for ir in range(len(px_ref_list)):
                    robot_future_x.append(px_ref_list[ir].unsqueeze(1))
                    robot_future_z.append(pz_ref_list[ir].unsqueeze(1))                    
                robot_future = torch.cat((torch.cat(robot_future_x, axis=1), torch.cat(robot_future_z, axis=1)), axis=1).to(device)  
                              
                ped_past_c = torch.clamp(ped_past, min=-10.0, max=10.0)/norm
                robot_past_c = torch.clamp(robot_past, min=-10.0, max=10.0)/norm
                robot_future_c = torch.clamp(robot_future, min=-10.0, max=10.0)/norm          
                flag_ped = (ped_past_c[:,0] != 0.0) & (ped_past_c[:,8] != 0.0)
                
                robot_past_c = 0.333*robot_past_c #frodbot is much faster than Vizbot.
                delta_est_ped_traj = model_pedtraj(ped_past_c, robot_past_c, robot_future_c)
                delta_est_ped_traj_zero = model_pedtraj(ped_past_c, robot_past_c, 0.0*robot_future_c)

                traj_x = torch.cumsum(delta_est_ped_traj[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_y = torch.cumsum(delta_est_ped_traj[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj = torch.clamp(torch.cat((traj_x, traj_y), axis=1), min=-10.0, max=10.0)

                delta_est_ped_traj_zero_detach = delta_est_ped_traj_zero.detach()
                traj_xz = torch.cumsum(delta_est_ped_traj_zero_detach[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_yz = torch.cumsum(delta_est_ped_traj_zero_detach[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj_zeros = torch.clamp(torch.cat((traj_xz, traj_yz), axis=1), min=-10.0, max=10.0)
                
                social_loss = nn.functional.mse_loss(est_ped_traj[flag_ped*combined_goal_mask], est_ped_traj_zeros[flag_ped*combined_goal_mask])
                
                max_pl = rsize.squeeze(1).repeat(1,8) + 0.5
                min_pl = rsize.squeeze(1).repeat(1,8) * 0.0
                personal_loss = nn.functional.mse_loss(max_pl[flag_ped*combined_goal_mask], torch.clamp(torch.sqrt((norm*robot_future_c[flag_ped*combined_goal_mask][:,0:8] - norm*est_ped_traj[flag_ped*combined_goal_mask][:,0:8])**2 + (norm*robot_future_c[flag_ped*combined_goal_mask][:,8:16] - norm*est_ped_traj[flag_ped*combined_goal_mask][:,8:16])**2), min=min_pl[flag_ped*combined_goal_mask], max= max_pl[flag_ped*combined_goal_mask]))
            else:
                est_ped_traj = torch.zeros(Bf+Bg, 16)
                est_ped_traj_zeros = torch.zeros(Bf+Bg, 16)      
                ped_past_c = torch.zeros(Bf+Bg, 16)
                robot_past_c = torch.zeros(Bf+Bg, 16)
                social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
            """
            est_ped_traj = torch.zeros(Bf+Bg, 16)
            est_ped_traj_zeros = torch.zeros(Bf+Bg, 16)      
            ped_past_c = torch.zeros(Bf+Bg, 16)
            robot_past_c = torch.zeros(Bf+Bg, 16)
            social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
            personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                            
            PC3D = []
            for j in range(len_traj_pred):
                px_ref = px_ref_list[6+j]
                pz_ref = pz_ref_list[6+j]
                ry_ref = ry_ref_list[6+j]                

                Tod = torch.zeros((Bf+Bg, 4, 4)).to(device)
                Tod[:, 0, 0] = torch.cos(ry_ref)
                Tod[:, 0, 2] = torch.sin(ry_ref)
                Tod[:, 1, 1] = 1.0
                Tod[:, 2, 0] = -torch.sin(ry_ref)
                Tod[:, 2, 2] = torch.cos(ry_ref)
                Tod[:, 0, 3] = px_ref
                Tod[:, 2, 3] = pz_ref
                Tod[:, 3, 3] = 1.0

                Ttrans = torch.inverse(Tod)[:, :3, :]               
                batch_3d_point_x = torch.cat((batch_3d_point.view(Bf+Bg, 3, -1), torch.ones(Bf+Bg,1,416*128).to(device)), axis=1)
                cam_points_trans = torch.matmul(Ttrans, batch_3d_point_x).view(Bf+Bg, 3, 128, 416)
                PC3D.append(cam_points_trans.unsqueeze(1))                                                  
            
            PC3D_cat = torch.cat(PC3D, axis=1)                    
            loss_geo = geometry_criterion_range(PC3D_cat[combined_goal_mask], rsize[:,:,0][combined_goal_mask], len_traj_pred, combined_geoloss_range[combined_goal_mask], device)
            loss = 4.0*dist_loss + 0.4*distall_loss + 0.5*diff_loss + 10.0*loss_geo + 100.0*social_loss + 10.0*personal_loss + 0.001*dist_temp_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"distall_loss": distall_loss.item()})            
            wandb.log({"smooth_loss": diff_loss.item()})
            wandb.log({"geo_loss": loss_geo.item()})
            wandb.log({"social_loss": social_loss.item()})
            wandb.log({"personal_loss": personal_loss.item()})
            wandb.log({"disttemp_loss": dist_temp_loss.item()}) 
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
                current_lrs = get_current_lr(optimizer)  
                print(i, current_lrs) 
            if i % 20000 == 0 and i != 0:
                lr_scheduler.step()
                current_lrs = get_current_lr(optimizer)  
                print(i, current_lrs) 
                
            if i % 5000 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss.item()
                losses['distall_loss'] = distall_loss.item()                
                losses['smooth_loss'] = diff_loss.item()                 
                losses['geo_loss'] = loss_geo.item()
                losses['social_loss'] = social_loss.item()                 
                losses['personal_loss'] = personal_loss.item()  
                losses['disttemp_loss'] = dist_temp_loss.item()  
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_exaug_delay_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    batch_3d_point,
                    goal_pos,
                    local_yaw,
                    linear_vel_d.cpu(),
                    angular_vel_d.cpu(),
                    norm*ped_past_c.cpu(),
                    norm*est_ped_traj.cpu(),
                    norm*est_ped_traj_zeros.cpu(),
                    norm*robot_past_c.cpu(),
                    last_poses.cpu(),
                    rsize.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )

###
def train_il_dist_gnm(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    #model_depth,
    #model_pedtraj,
    #device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,                
    }       
    dataloader_sub_iter = iter(dataloader_sub)           
       
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                which_dataset,
                obs_image_future,
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data         
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    dataset_index_sub,
                    action_mask_sub,
                ) = next(dataloader_sub_iter)        
    
            Bsub, _, H, W = obs_image_sub.size()  

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE)
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)
            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
                    
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break            
            B, _, H, W = obs_image.size()  
            
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)                  
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)
            batch_goal_pos = goal_pos.to(device)
        
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions = torch.cat((actions, action_label_sub), axis=0)   
             
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 

            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)                                            
            combined_dist_pred, combined_action_pred = model(combined_obs_image, combined_goal_image)   
                                
            losses = _compute_losses(
                dist_label=combined_distance,
                action_label=combined_actions,
                dist_pred=combined_dist_pred,
                action_pred=combined_action_pred,
                alpha=0.5,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 20000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 5000 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il_estimation(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )               
                                     
###
def train_il2_dist_gnm_gps(
    model: nn.Module,
    model_GNM: nn.Module,    
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                
    }     
    dataloader_sub_iter = iter(dataloader_sub)           
    
    model_GNM.eval().to(device)        
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data
            
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)        
    
            Bsub, _, H, W = obs_image_sub.size()  

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])        
            goal_image_sub = transform(goal_image_sub).to(device)
            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
                    
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break
            
            B, _, H, W = obs_image.size()              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            batch_goal_pos = goal_pos.to(device)       
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
             
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub), axis=0).to(device) 
            
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)                                            
            combined_action_pred = model(combined_obs_image, combined_goal_pos_gps)   

            #labeling by Robotic foundation model (IL on GNM dataset)
            with torch.no_grad():
                dist_estfrod, action_estfrod = model_GNM(obs_image, goal_image)                   
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)    
                                                  
            losses = _compute_losses_gps(
                action_label=combined_actions_origin.to(device),                
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )      
###
def train_il_exaug_dist_gnm_gps(
    model: nn.Module,
    model_GNM: nn.Module,    
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,  
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                
    }          
    dataloader_sub_iter = iter(dataloader_sub)           
    
    model_GNM.eval().to(device)          
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                obs_image_crop,                 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,
            ) = data

            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)        
    
            Bsub, _, H, W = obs_image_sub.size()  

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image_sub = transform(goal_image_sub).to(device)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)

            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break
            
            B, _, H, W = obs_image.size()  
            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images_crop = torch.split(obs_image_crop, 3, dim=1)            
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_images_crop = [transform(obs_image_crop).to(device) for obs_image_crop in obs_images_crop]            
            obs_image = torch.cat(obs_images, dim=1)
            obs_image_crop = torch.cat(obs_images_crop, dim=1)            
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            batch_goal_pos = goal_pos.to(device)    
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_obs_image_crop = torch.cat((obs_image_crop, obs_image_sub), axis=0)            
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
            
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub), axis=0).to(device) 
            
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)                                            
            combined_action_pred = model(combined_obs_image_crop, combined_goal_pos_gps)   

            rsize = 0.3*torch.ones(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B, 1, 1).to(device)   
            linear_vel_old = 0.5*torch.ones(B, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
             
            with torch.no_grad():
                linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
            
            
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
                                                  
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            if losses["total_loss"].item() < 20:
                optimizer.zero_grad()
                losses["total_loss"].backward()
                optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )      
###
def train_il_exaug_dist_gnm_gps_map(
    model: nn.Module,
    model_GNM: nn.Module,    
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool, 
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    

    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                
    }
    """
    mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device)
    """           
    dataloader_sub_iter = iter(dataloader_sub)           
    
    model_GNM.eval().to(device)      
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            
            (
                obs_image, 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                current_map_image,
                goal_map_image,
                ped_list_no_trans,
                robot_list,
            ) = data
            
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)           
            Bsub, _, H, W = obs_image_sub.size()  

            current_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            goal_map_image_sub = torch.zeros(Bsub, 3, 96, 96)

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub_map = obs_images_sub[-1].to(device)
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            map_images_sub = torch.cat((transform(current_map_image_sub).to(device), transform(goal_map_image_sub).to(device), obs_image_sub_map), axis=1)
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
                    
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break
            
            B, _, H, W = obs_image.size()  
            
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])                 
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image_map = obs_images[-1].to(device)
            map_images = torch.cat((transform(current_map_image).to(device), transform(goal_map_image).to(device), obs_image_map), axis=1)            
            
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)
            batch_goal_pos = goal_pos.to(device)        
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_goal_image = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)            
            combined_distance = torch.cat((distance, dist_label_sub), axis=0)   
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 
            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0)          
            combined_viz_cur_map = torch.cat((current_map_image, current_map_image_sub), axis=0)
            combined_viz_goal_map = torch.cat((goal_map_image, goal_map_image_sub), axis=0)                   
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub), axis=0).to(device) 
            combined_map_images = torch.cat((map_images, map_images_sub), axis=0)
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)     
            
            #model calculate 
            goal_mask_f = torch.randint(0,3,(B,)).to(device) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            goal_mask_sub = torch.zeros(Bsub).to(device) #GNM: Goal pose only
            goal_mask_select = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            
            combined_action_pred, mask_number = model(combined_obs_image, combined_goal_pos_gps, combined_map_images, goal_mask_select)   

            rsize = 0.3*torch.ones(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B, 1, 1).to(device)   
            linear_vel_old = 0.5*torch.ones(B, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
             
            with torch.no_grad():
                linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
                     
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
                                                  
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )            
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()                        
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu             
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation_map(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_viz_cur_map,
                    combined_viz_goal_map,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    mask_number,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )      

def train_il_exaug_dist_gnm_gps_map2(
    model: nn.Module,
    model_GNM: nn.Module,    
    ema_model: EMAModel,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    #model_depth,
    #model_pedtraj,
    #device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    
    action_loss_logger_0 = Logger("action_loss_0", "train", window_size=print_log_freq)    
    action_loss_logger_1 = Logger("action_loss_1", "train", window_size=print_log_freq)    
    action_loss_logger_2 = Logger("action_loss_2", "train", window_size=print_log_freq)    
    action_loss_logger_3 = Logger("action_loss_3", "train", window_size=print_log_freq)    
    action_loss_logger_4 = Logger("action_loss_4", "train", window_size=print_log_freq)    
    action_loss_logger_5 = Logger("action_loss_5", "train", window_size=print_log_freq)    
    action_loss_logger_6 = Logger("action_loss_6", "train", window_size=print_log_freq)        
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger, 
        "action_loss_0": action_loss_logger_0, 
        "action_loss_1": action_loss_logger_1, 
        "action_loss_2": action_loss_logger_2, 
        "action_loss_3": action_loss_logger_3, 
        "action_loss_4": action_loss_logger_4, 
        "action_loss_5": action_loss_logger_5, 
        "action_loss_6": action_loss_logger_6,                
    }
    
    #D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    #K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))
    #
    #mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    #mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    #mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    #           
    dataloader_sub_iter = iter(dataloader_sub)           
    
    map_image_gen = MapTileCache(path_mapcache + "/map_tiles_satellite")
    transform_PIL_tensor = transforms.ToTensor()
    #ema_model.eval()
    
    model_GNM.eval().to(device)
    #model2 = copy.deepcopy(model).to(device2)            
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                obs_image_crop,                 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                current_map_image,
                goal_map_image,
                ped_list_no_trans,
                robot_list,
                #lat,
                #lon,
                #compass,
                #lat_cur,
                #lon_cur,
                #compass_cur,                
            ) = data
            
            #gc.collect()
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)        
            #print(current_map_image.size(), goal_map_image.size())
     
            #print(lat.size(), lon.size(), compass.size())
            """
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item()).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))
            """
            """
            lat_dummy = 37.87370638591221
            lon_dummy =  -122.26739537451519
            compass_dummy = 0.0
            
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_dummy, lon_dummy, compass_dummy).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_ViNTLeRobotDataset_IL2_gps_map_cropposition(lat_dummy, lon_dummy, compass_dummy, 50.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))            
            """
            """
            #print("before", lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item())
            #print("before", lat_dummy, lon_dummy, compass_dummy)
            #print("after", new_lat_1, new_lon_1, new_heading_1)
            #print("delta", new_lat_1-lat_cur[0].item(), new_lon_1-lon_cur[0].item())
            #print("delta", new_lat_1-lat_dummy, new_lon_1-lon_dummy)            
            #current_map_image[1] = current_map_image[0]
            current_map_image[1] = cur_map_gen1
            current_map_image[2] = current_map_image[0]
            current_map_image[3] = current_map_image[0]
            #goal_map_image[1] = goal_map_image[0]       

            new_lat_2, new_lon_2, new_heading_2 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 10.0, -30.0/180*3.1415)
            goal_map_gen2 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_2, new_lon_2, new_heading_2).resize((96,96), resample=Image.Resampling.LANCZOS))
            new_lat_3, new_lon_3, new_heading_3 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, -10.0, +30.0/180*3.1415)
            goal_map_gen3 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_3, new_lon_3, new_heading_3).resize((96,96), resample=Image.Resampling.LANCZOS))
                                         
            goal_map_image[1] = goal_map_gen1               
            goal_map_image[2] = goal_map_gen2  
            goal_map_image[3] = goal_map_gen3                                           
            """
            Bsub, _, H, W = obs_image_sub.size()  
            current_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            goal_map_image_sub = torch.zeros(Bsub, 3, 96, 96)

            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub_map = obs_images_sub[-1].to(device)
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            map_images_sub = torch.cat((transform(current_map_image_sub).to(device), transform(goal_map_image_sub).to(device), obs_image_sub_map), axis=1)
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)
            #model_outputs = model(obs_image, goal_image)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
            #print("goal_pose_gps_sub", goal_pose_gps_sub.size())
                        
            #print("data sub", obs_image_sub.size(), action_label_sub.size(), goal_mask_sub.sum())   
                    
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break
                
            """
            current_image_depth = (current_image.to(device2))*mask_360_torch
            B, _, H, W = current_image_depth.size()    
            
            with torch.no_grad():
                #depth estimation
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device)   
            """
            
            B, _, H, W = obs_image.size()  
            
            #print("batch size", B, Bsub)
            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images_crop = torch.split(obs_image_crop, 3, dim=1)            
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            #obs_image_map = obs_images[-1].to(device)
            obs_images_crop = [transform(obs_image).to(device) for obs_image in obs_images_crop]
            obs_image_map_crop = obs_images_crop[-1].to(device)                                      
                                             
            #obs_image_map_crop[1] = obs_image_map_crop[0]
            #obs_image_map_crop[2] = obs_image_map_crop[0]
            #obs_image_map_crop[3] = obs_image_map_crop[0]                                
                                                                          
            map_images = torch.cat((transform(current_map_image).to(device), transform(goal_map_image).to(device), obs_image_map_crop), axis=1)            
            
            obs_image = torch.cat(obs_images, dim=1)
            obs_image_crop = torch.cat(obs_images_crop, dim=1)            
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            #obs_image_crop[1] = obs_image_crop[0]
            #obs_image_crop[2] = obs_image_crop[0]
            #obs_image_crop[3] = obs_image_crop[0]      

            batch_goal_pos = goal_pos.to(device)
        
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
            #print("goal_pose_gps", goal_pose_gps.size())
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        
            #print(goal_mask)

            """
            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]
                    #print(ig, igr)
            """
            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            #print(obs_image.size(), obs_image_sub.size())
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_obs_image_crop = torch.cat((obs_image_crop, obs_image_sub), axis=0)
            combined_goal_image_crop = torch.cat((goal_image, goal_image_sub), axis=0)            
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
            
            #print(distance.mean(), dist_label_sub.mean())      
            #print("distance", distance)
            #print("dist_label_sub", dist_label_sub)model2   
            #print(distance.max(), distance.min(), dist_label_sub.max(), dist_label_sub.min())
            combined_distance = torch.clip(torch.cat((distance, dist_label_sub), axis=0), min=0.0, max=20.0) 
            combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            
            combined_viz_cur_map = torch.cat((current_map_image, current_map_image_sub), axis=0)
            combined_viz_goal_map = torch.cat((goal_map_image, goal_map_image_sub), axis=0)            
            
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub), axis=0).to(device) 
            combined_map_images = torch.cat((map_images, map_images_sub), axis=0)
            #print(local_yaw.size())
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)     
            
            goal_mask_frod = []
            goal_mask_gnm = []
            for idf in range(B):
                if distance[idf] <= 20:
                    goal_mask_frod.append(random.randint(0,6))
                    #goal_mask_frod.append(0)
                else:
                    goal_mask_frod.append(random.randint(0,5))
                    #goal_mask_frod.append(0)
                    
            for idg in range(Bsub):
                if dist_label_sub[idg] <= 20:
                    goal_mask_gnm.append(random.randint(4,6))
                else:
                    goal_mask_gnm.append(random.randint(4,5))
                                                        
            """                                            
            goal_mask_f = torch.randint(0,3,(B,)).to(device) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            #goal_mask_sub = torch.zeros(Bsub).to(device) #GNM: Goal pose only
            goal_mask_sub = torch.ones(Bsub).to(device) #GNM: Goal pose only            
            goal_mask_select = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            """
            goal_mask_select = torch.tensor(goal_mask_frod + goal_mask_gnm).to(device)            
            combined_action_pred, combined_dist_pred, mask_number = model(combined_obs_image_crop, combined_goal_pos_gps, combined_map_images, combined_goal_image_crop, goal_mask_select)   
            #print(combined_action_pred.mean(), combined_dist_pred.mean(), combined_distance.mean(), combined_dist_pred.size(), combined_distance.size())
            
            #print(combined_dist_pred.mean(), combined_dist_pred.max(), combined_dist_pred.min())
            #print(combined_distance.mean(), combined_distance.max(), combined_distance.min())
            
            #print("pred", combined_dist_pred)
            #print("gt", combined_distance)            
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            with torch.no_grad():
                dist_estfrod, action_estfrod = model_GNM(obs_image, goal_image)   
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image, combined_goal_image)  
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image.to(device2), combined_goal_image.to(device2))  
                #dist_estfrod, combined_actions = model(combined_obs_image, combined_goal_image)                  
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
            #with torch.no_grad():
            #    dist_estfrod, combined_actions = model2(combined_obs_image.to(device2), combined_goal_image.to(device2))   
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            rsize = 0.3*torch.ones(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B, 1, 1).to(device)   
            linear_vel_old = 0.5*torch.ones(B, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
             
            with torch.no_grad():
                #linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image, rsize, delay, vel_past)
                linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
            
            
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            #print(x_traj_cat.size(), z_traj_cat.size(), yaw_traj_cat.size())
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
            """                                   
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )
            """
            loss_list = []
            for icl in range(7):
                mask_task = goal_mask_select == icl
                #print(mask_task)
                losses = _compute_losses(
                    dist_label=combined_distance[mask_task].to(device),
                    action_label=combined_actions[mask_task].to(device),
                    dist_pred=combined_dist_pred[mask_task],
                    action_pred=combined_action_pred[mask_task],
                    alpha=0.5,
                    learn_angle=True,
                    action_mask=combined_action_mask[mask_task],
                )   
                loss_list.append(losses)
                action_loss_cpu = losses["action_loss"].item()                 
                wandb.log({"action_loss_" + str(icl): action_loss_cpu}) 
                        
            losses = _compute_losses(
                dist_label=combined_distance.to(device),
                action_label=combined_actions.to(device),
                dist_pred=combined_dist_pred,
                action_pred=combined_action_pred,
                alpha=0.5,
                learn_angle=True,
                action_mask=combined_action_mask,
            )
                        
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
                          
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)
                    #numbered_path = os.path.join(project_folder, f"ema_latest.pth")

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                for icl in range(7):
                    losses['action_loss_' + str(icl)] = loss_list[icl]['action_loss'].item()    
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    #print(key)
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation_map2(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_viz_cur_map,
                    combined_viz_goal_map,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    mask_number,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )     

def train_il_exaug_dist_gnm_gps_map2_lan(
    model: nn.Module,
    model_GNM: nn.Module,    
    text_encoder: nn.Module,    
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,
    noise_scheduler: DDPMScheduler,    
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    dataloader_lan: DataLoader,       
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    #model_depth,
    #model_pedtraj,
    #device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    lan_solo: bool = False,  
    image_solo: bool = False,  
    sate_solo: bool = False,  
    no_sate: bool = False,   
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()  

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)   
    obj_loss_logger = Logger("obj_loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth_loss", "train", window_size=print_log_freq)      
    action_loss_logger_0 = Logger("action_loss_0", "train", window_size=print_log_freq)    
    action_loss_logger_1 = Logger("action_loss_1", "train", window_size=print_log_freq)    
    action_loss_logger_2 = Logger("action_loss_2", "train", window_size=print_log_freq)    
    action_loss_logger_3 = Logger("action_loss_3", "train", window_size=print_log_freq)    
    action_loss_logger_4 = Logger("action_loss_4", "train", window_size=print_log_freq)    
    action_loss_logger_5 = Logger("action_loss_5", "train", window_size=print_log_freq)    
    action_loss_logger_6 = Logger("action_loss_6", "train", window_size=print_log_freq)    
    action_loss_logger_7 = Logger("action_loss_7", "train", window_size=print_log_freq)    
    action_loss_logger_8 = Logger("action_loss_8", "train", window_size=print_log_freq)         
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger, 
        "obj_loss": obj_loss_logger,
        "smooth_loss": smooth_loss_logger,         
        "action_loss_0": action_loss_logger_0, 
        "action_loss_1": action_loss_logger_1, 
        "action_loss_2": action_loss_logger_2, 
        "action_loss_3": action_loss_logger_3, 
        "action_loss_4": action_loss_logger_4, 
        "action_loss_5": action_loss_logger_5, 
        "action_loss_6": action_loss_logger_6,     
        "action_loss_7": action_loss_logger_7, 
        "action_loss_8": action_loss_logger_8,                    
    }
    
    #D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    #K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))
    #
    #mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    #mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    #mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    #           
    dataloader_sub_iter = iter(dataloader_sub)           
    dataloader_lan_iter = iter(dataloader_lan)   
        
    map_image_gen = MapTileCache(path_mapcache + "/map_tiles_satellite")
    transform_PIL_tensor = transforms.ToTensor()
    #ema_model.eval() 
    text_encoder.eval().to(device)
    
    model_GNM.eval().to(device)
    #model2 = copy.deepcopy(model).to(device2)            
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                obs_image_crop,                 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                current_map_image,
                goal_map_image,
                ped_list_no_trans,
                robot_list,
                #lat,
                #lon,
                #compass,
                #lat_cur,
                #lon_cur,
                #compass_cur,                
            ) = data
            
            #gc.collect()
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)    
            try:
                (
                    obs_images_lan, 
                    goal_image_lan,
                    cur_large_img_lan,
                    goal_pos_lan,
                    obj_inst_lan,
                    goal_pos_norm_lan,    
                    goal_image_full_lan,
                    goal_image_full_8_lan,                    
                    distance_lan,            
                    action_mask_lan,
                ) = next(dataloader_lan_iter)
            except StopIteration:
                dataloader_lan_iter = iter(dataloader_lan) 
                (
                    obs_images_lan, 
                    goal_image_lan,
                    cur_large_img_lan,
                    goal_pos_lan,
                    obj_inst_lan,
                    goal_pos_norm_lan,     
                    goal_image_full_lan,
                    goal_image_full_8_lan,                    
                    distance_lan,  
                    action_mask_lan,                                                   
                ) = next(dataloader_lan_iter)  
                        
            Blan, _, H, W = obs_images_lan.size()             
            current_map_image_lan = torch.zeros(Blan, 3, 96, 96)
            goal_map_image_lan = torch.zeros(Blan, 3, 96, 96)            
            goal_images_crop = transform(goal_image_lan).to(device)
            goal_image2_lan = transform(goal_image_full_8_lan).to(device) 
            
            obs_images_lan_list = torch.split(obs_images_lan, 3, dim=1)
            #curobs_image_lan = obs_images_lan_list[-1]              
            
            batch_viz_obs_images_lan = TF.resize((255.0*obs_images_lan).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images_lan = TF.resize((255.0*goal_image_lan).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])

            obs_images_lan = [transform(obs_image_lan).to(device) for obs_image_lan in obs_images_lan_list]
            obs_image_lan_map = obs_images_lan_list[-1]  
            
            obs_image_lan = torch.cat(obs_images_lan, dim=1)
            obs_image_lan_nomad = torch.cat(obs_images_lan[-2:], dim=1)
            goal_pos_norm_lan = goal_pos_norm_lan.to(device) 
                            
            cur_large_img_lan = transform(cur_large_img_lan).to(device)
            #, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)
            
            dis_obj = torch.sqrt(goal_pos_lan[:,1:2]**2 + goal_pos_lan[:,0:1]**2)
            #print(dis_obj.size(), (goal_pos_lan[:,1:2]/dis_obj).size())
            goal_pose_gps_lan = torch.cat((goal_pos_lan[:,1:2], -goal_pos_lan[:,0:1], goal_pos_lan[:,1:2]/dis_obj, -goal_pos_lan[:,0:1]/dis_obj), axis=1)
            #print(goal_pos_lan.size(), goal_pose_gps_lan.size())
            map_images_lan = torch.cat((transform(current_map_image_lan).to(device), transform(goal_map_image_lan).to(device), obs_image_lan_map.to(device)), axis=1)
            goal_image_lan = transform(goal_image_full_lan).to(device)
            distance_lan = distance_lan.float().to(device) 
            action_mask_lan = action_mask_lan.to(device)
                        
            batch_obj_inst_lan = clip.tokenize(obj_inst_lan, truncate=True).to(device) 
            with torch.no_grad():  
                feat_text_lan = text_encoder.encode_text(batch_obj_inst_lan)            
                
            #print("feat_text", feat_text_lan.size())
            #print("batch_obs_images_lan", cur_large_img_lan.size())  
            
            with torch.no_grad():
                select_traj = supervision_from_nomad(
                    ema_model_nomad,
                    noise_scheduler,
                    obs_image_lan_nomad,
                    goal_images_crop,
                    batch_viz_obs_images_lan,
                    batch_viz_goal_images_lan,
                    goal_pos_norm_lan,
                    device,
                    project_folder,
                    epoch,
                    Blan,
                    i,                
                    30,
                    use_wandb,
                    )               
            
            """
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item()).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))
            """
            """
            lat_dummy = 37.87370638591221
            lon_dummy =  -122.26739537451519
            compass_dummy = 0.0
            
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_dummy, lon_dummy, compass_dummy).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_ViNTLeRobotDataset_IL2_gps_map_cropposition(lat_dummy, lon_dummy, compass_dummy, 50.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))            
            """
            """
            #print("before", lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item())
            #print("before", lat_dummy, lon_dummy, compass_dummy)
            #print("after", new_lat_1, new_lon_1, new_heading_1)
            #print("delta", new_lat_1-lat_cur[0].item(), new_lon_1-lon_cur[0].item())
            #print("delta", new_lat_1-lat_dummy, new_lon_1-lon_dummy)            
            #current_map_image[1] = current_map_image[0]
            current_map_image[1] = cur_map_gen1
            current_map_image[2] = current_map_image[0]
            current_map_image[3] = current_map_image[0]
            #goal_map_image[1] = goal_map_image[0]       

            new_lat_2, new_lon_2, new_heading_2 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 10.0, -30.0/180*3.1415)
            goal_map_gen2 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_2, new_lon_2, new_heading_2).resize((96,96), resample=Image.Resampling.LANCZOS))
            new_lat_3, new_lon_3, new_heading_3 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, -10.0, +30.0/180*3.1415)
            goal_map_gen3 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_3, new_lon_3, new_heading_3).resize((96,96), resample=Image.Resampling.LANCZOS))
                                         
            goal_map_image[1] = goal_map_gen1               
            goal_map_image[2] = goal_map_gen2  
            goal_map_image[3] = goal_map_gen3                                           
            """
            Bsub, _, H, W = obs_image_sub.size()  
            current_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            goal_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            cur_large_img_sub = torch.zeros(Bsub, 3, 224, 224).to(device)  
            feat_text_sub = torch.zeros(Bsub, 512).to(device)  
             
            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub_map = obs_images_sub[-1].to(device)
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            map_images_sub = torch.cat((transform(current_map_image_sub).to(device), transform(goal_map_image_sub).to(device), obs_image_sub_map), axis=1)
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)
            #model_outputs = model(obs_image, goal_image)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
            #print("goal_pose_gps_sub", goal_pose_gps_sub.size())
                        
            #print("data sub", obs_image_sub.size(), action_label_sub.size(), goal_mask_sub.sum())   
                    
            #if psutil.virtual_memory().percent > 90.0:
            #    print("RAM usage (%)", psutil.virtual_memory().percent)
            #    break
                
            """
            current_image_depth = (current_image.to(device2))*mask_360_torch
            B, _, H, W = current_image_depth.size()    
            
            with torch.no_grad():
                #depth estimation
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device)   
            """
            
            B, _, H, W = obs_image.size()  
            
            #print("batch size", B, Bsub)
            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images_crop = torch.split(obs_image_crop, 3, dim=1)            
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            cur_large_img = torch.zeros(B, 3, 224, 224).to(device)  
            feat_text = torch.zeros(B, 512).to(device)  
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            #obs_image_map = obs_images[-1].to(device)
            obs_images_crop = [transform(obs_image).to(device) for obs_image in obs_images_crop]
            obs_image_map_crop = obs_images_crop[-1].to(device)                                      
                                             
            #obs_image_map_crop[1] = obs_image_map_crop[0]
            #obs_image_map_crop[2] = obs_image_map_crop[0]
            #obs_image_map_crop[3] = obs_image_map_crop[0]  
            """  
            print("current_map_images max min", transform(current_map_image).max(), transform(current_map_image).min(), current_map_image.max(), current_map_image.min())                            
            print("current_map_images mean", torch.mean(transform(current_map_image).to(device)[:,0]), torch.mean(transform(current_map_image).to(device)[:,1]), torch.mean(transform(current_map_image).to(device)[:,2])) 
            print("current map_images std", torch.std(transform(current_map_image).to(device)[:,0]), torch.std(transform(current_map_image).to(device)[:,1]), torch.std(transform(current_map_image).to(device)[:,2]))  
            print("goal_map_images max min", transform(goal_map_image).max(), transform(goal_map_image).min(), goal_map_image.max(), goal_map_image.min())               
            print("goal_map_images mean", torch.mean(transform(goal_map_image).to(device)[:,0]), torch.mean(transform(goal_map_image).to(device)[:,1]), torch.mean(transform(goal_map_image).to(device)[:,2])) 
            print("goal_map_images std", torch.std(transform(goal_map_image).to(device)[:,0]), torch.std(transform(goal_map_image).to(device)[:,1]), torch.std(transform(goal_map_image).to(device)[:,2])) 
            print("obs_image_map_crop max min", obs_image_map_crop.max(), obs_image_map_crop.min())                   
            print("obs_image_map_crop mean", torch.mean(obs_image_map_crop.to(device)[:,0]), torch.mean(obs_image_map_crop.to(device)[:,1]), torch.mean(obs_image_map_crop.to(device)[:,2])) 
            print("obs_image_map_crop std", torch.std(obs_image_map_crop.to(device)[:,0]), torch.std(obs_image_map_crop.to(device)[:,1]), torch.std(obs_image_map_crop.to(device)[:,2]))        
            """                                                                                              
            map_images = torch.cat((transform(current_map_image).to(device), transform(goal_map_image).to(device), obs_image_map_crop), axis=1)            
            
            obs_image = torch.cat(obs_images, dim=1)
            obs_image_crop = torch.cat(obs_images_crop, dim=1)            
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            #obs_image_crop[1] = obs_image_crop[0]
            #obs_image_crop[2] = obs_image_crop[0]
            #obs_image_crop[3] = obs_image_crop[0]      

            batch_goal_pos = goal_pos.to(device)
        
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
            #print("goal_pose_gps", goal_pose_gps.size())
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        
            #print(goal_mask)

            """
            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]
                    #print(ig, igr)
            """
            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            #print(obs_image.size(), obs_image_sub.size())
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_obs_image_crop = torch.cat((obs_image_crop, obs_image_sub, obs_image_lan), axis=0)
            combined_goal_image_crop = torch.cat((goal_image, goal_image_sub, goal_image_lan), axis=0)            
            combined_cur_large_img = torch.cat((cur_large_img, cur_large_img_sub, cur_large_img_lan), axis=0)    
            combined_feat_text = torch.cat((feat_text, feat_text_sub, feat_text_lan), axis=0)  
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
            
            #print(distance.mean(), dist_label_sub.mean())      
            #print("distance", distance)
            #print("dist_label_sub", dist_label_sub)model2   
            #print(distance.max(), distance.min(), dist_label_sub.max(), dist_label_sub.min())
            combined_distance = torch.clip(torch.cat((distance, dist_label_sub, distance_lan), axis=0), min=0.0, max=20.0) 
            #combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub, action_mask_lan), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            
            combined_viz_cur_map = torch.cat((current_map_image, current_map_image_sub), axis=0)
            combined_viz_goal_map = torch.cat((goal_map_image, goal_map_image_sub), axis=0)            
            
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub, goal_pose_gps_lan), axis=0).to(device) 
            
            combined_map_images = torch.cat((map_images, map_images_sub, map_images_lan), axis=0)
            #print(local_yaw.size())
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)     
            
            goal_mask_frod = []
            goal_mask_gnm = []
            goal_mask_lan = []    
            
            if image_solo == True:
                for idf in range(B):
                    goal_mask_frod.append(6)
                for idg in range(Bsub):
                    goal_mask_gnm.append(6)
                for idl in range(Blan):    
                    goal_mask_lan.append(6)
            else:
                if no_sate == True:            
                    for idf in range(B):
                        if sate_solo:
                            goal_mask_frod.append(0)
                        else:
                            if distance[idf] <= 20:
                                goal_mask_frod.append(random.randint(4,6))
                                #goal_mask_frod.append(0)
                            else:
                                goal_mask_frod.append(random.randint(4,5))
                                #goal_mask_frod.append(0)
                else:            
                    for idf in range(B):
                        if sate_solo:
                            goal_mask_frod.append(0)
                        else:
                            if distance[idf] <= 20:
                                goal_mask_frod.append(random.randint(0,6))
                                #goal_mask_frod.append(0)
                            else:
                                goal_mask_frod.append(random.randint(0,5))
                                #goal_mask_frod.append(0)
                                                
                for idg in range(Bsub):
                    if dist_label_sub[idg] <= 20:
                        goal_mask_gnm.append(random.randint(4,6))
                    else:
                        goal_mask_gnm.append(random.randint(4,5))
                    
                #my_list = [4,7,8]
                my_list = [7,8]            
                for idl in range(Blan):
                    if distance_lan[idl] == 0:
                        #goal_mask_lan.append(random.choice(my_list)) 
                        if lan_solo: 
                            goal_mask_lan.append(7)
                        else:
                            if random.random() > 0.5:
                                goal_mask_lan.append(7)
                            else:
                                goal_mask_lan.append(random.choice(my_list))   
                    else:
                        #goal_mask_lan.append(random.randint(6,7))                
                        goal_mask_lan.append(6)      
                                                                            
            """                                            
            goal_mask_f = torch.randint(0,3,(B,)).to(device) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            #goal_mask_sub = torch.zeros(Bsub).to(device) #GNM: Goal pose only
            goal_mask_sub = torch.ones(Bsub).to(device) #GNM: Goal pose only            
            goal_mask_select = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            """
            goal_mask_select = torch.tensor(goal_mask_frod + goal_mask_gnm + goal_mask_lan).to(device)            
            combined_action_pred, combined_dist_pred, mask_number = model(combined_obs_image_crop, combined_goal_pos_gps, combined_map_images, combined_goal_image_crop, goal_mask_select, combined_feat_text, combined_cur_large_img)   
            #print(combined_action_pred.mean(), combined_dist_pred.mean(), combined_distance.mean(), combined_dist_pred.size(), combined_distance.size())
            
            #print(combined_dist_pred.mean(), combined_dist_pred.max(), combined_dist_pred.min())
            #print(combined_distance.mean(), combined_distance.max(), combined_distance.min())
            
            #print("pred", combined_dist_pred)
            #print("gt", combined_distance)            
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            with torch.no_grad():
                dist_estfrod, action_estfrod = model_GNM(obs_image, goal_image)   
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image, combined_goal_image)  
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image.to(device2), combined_goal_image.to(device2))  
                #dist_estfrod, combined_actions = model(combined_obs_image, combined_goal_image)                  
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
            #with torch.no_grad():
            #    dist_estfrod, combined_actions = model2(combined_obs_image.to(device2), combined_goal_image.to(device2))   
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            rsize = 0.3*torch.ones(B + Blan, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B + Blan, 1, 1).to(device)   
            linear_vel_old = 0.5*torch.ones(B + Blan, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B + Blan, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
            
            obs_image_mbra = torch.cat((obs_image, obs_image_lan), axis=0)
            goal_image_mbra = torch.cat((goal_image2, goal_image2_lan), axis=0)
             
            with torch.no_grad():
                #linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image_mbra, goal_image_mbra, rsize, delay, vel_past)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
            
            
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            #print(x_traj_cat.size(), z_traj_cat.size(), yaw_traj_cat.size())
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            
            mask_0 = (distance_lan == 0).float().unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            mask_non0 = (distance_lan != 0).float().unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            
            #print(select_traj.size(), action_estfrod[B:B+Blan].size())    
            select_traj_x1 = torch.cat((torch.zeros(Blan, 1, 1).to(device), select_traj[:,0:7,0:1]), axis=1)
            select_traj_x2 = select_traj[:,:,0:1]
            select_traj_y1 = torch.cat((torch.zeros(Blan, 1, 1).to(device), select_traj[:,0:7,1:2]), axis=1)
            select_traj_y2 = select_traj[:,:,1:2]          
            dist_select_traj = torch.sqrt((select_traj_x2 - select_traj_x1)**2 + (select_traj_y2 - select_traj_y1)**2)
            cos_select_traj = (select_traj_x2 - select_traj_x1)/dist_select_traj
            sin_select_traj = (select_traj_y2 - select_traj_y1)/dist_select_traj
            select_traj_ang = torch.cat((select_traj, cos_select_traj, sin_select_traj), axis=2)
            
            action_label_lan = mask_non0*action_estfrod[B:B+Blan].detach().to(device) + mask_0*select_traj_ang.detach().to(device)
            #print(action_label_lan)
            combined_actions = torch.cat((action_estfrod[0:B].detach().to(device), action_label_sub, action_label_lan), axis=0)   
            
            #print(goal_pos_lan)
            tar_obj_pose = (goal_pos_lan.to(device))/metric_waypoint_spacing
            action_pred_lan = combined_action_pred[B+Bsub:B+Bsub+Blan]
            #print(action_pred_lan.size(), action_pred_lan[:,-1,0:2].size(), tar_obj_pose.size())
            
            #print(B, Blan, action_estfrod[0:B].size(), action_label_sub.size(), action_estfrod[B:Blan].size())
            """                                   
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )
            """
            #frod_loss = nn.functional.mse_loss(combined_action_pred[0:B], combined_actions[0:B])
            #gnm_loss = nn.functional.mse_loss(combined_action_pred[B:B+Bsub], combined_actions[B:B+Bsub])           
            #lelan_loss = nn.functional.mse_loss(combined_action_pred[B+Bsub:], combined_actions[B+Bsub:])   
            #print(combined_action_pred.size(), frod_loss, gnm_loss, lelan_loss)            
            
            #print(goal_mask_select.size(), combined_distance.size(), combined_actions.size())
            loss_list = []
            for icl in range(9):
                mask_task = goal_mask_select == icl
                #print(mask_task)
                losses = _compute_losses(
                    dist_label=combined_distance[mask_task].to(device),
                    action_label=combined_actions[mask_task].to(device),
                    dist_pred=combined_dist_pred[mask_task],
                    action_pred=combined_action_pred[mask_task],
                    alpha=0.5,
                    learn_angle=True,
                    action_mask=combined_action_mask[mask_task],
                )   
                loss_list.append(losses)
                action_loss_cpu = losses["action_loss"].item()                 
                wandb.log({"action_loss_" + str(icl): action_loss_cpu}) 
            
            mask_lan = (distance_lan == 0)*((goal_mask_select == 7) + (goal_mask_select == 8))[B+Bsub:B+Bsub+Blan]
            if lan_solo:
                losses = _compute_losses_lan(
                    dist_label=combined_distance[B+Bsub:].to(device),
                    action_label=combined_actions[B+Bsub:].to(device),
                    dist_pred=combined_dist_pred[B+Bsub:],
                    action_pred=combined_action_pred[B+Bsub:],
                    pose_obj_label=tar_obj_pose[mask_lan],
                    pose_obj_pred=action_pred_lan[:,-1,0:2][mask_lan],
                    alpha=0.5,
                    learn_angle=True,
                    image_solo=image_solo,
                    sate_solo=sate_solo,                    
                    action_mask=combined_action_mask[B+Bsub:],
                )            
                #print(combined_distance[mask_lan].size(), combined_distance.size())
            elif sate_solo:
                losses = _compute_losses_lan(
                    dist_label=combined_distance[0:B].to(device),
                    action_label=combined_actions[0:B].to(device),
                    dist_pred=combined_dist_pred[0:B],
                    action_pred=combined_action_pred[0:B],
                    pose_obj_label=tar_obj_pose[mask_lan],
                    pose_obj_pred=action_pred_lan[:,-1,0:2][mask_lan],
                    alpha=0.5,
                    learn_angle=True,
                    image_solo=image_solo,
                    sate_solo=sate_solo,                    
                    action_mask=combined_action_mask[0:B],
                )            
                #print(combined_distance[mask_lan].size(), combined_distance.size())            
            else:
                losses = _compute_losses_lan(
                    dist_label=combined_distance.to(device),
                    action_label=combined_actions.to(device),
                    dist_pred=combined_dist_pred,
                    action_pred=combined_action_pred,
                    pose_obj_label=tar_obj_pose[mask_lan],
                    pose_obj_pred=action_pred_lan[:,-1,0:2][mask_lan],
                    alpha=0.5,
                    learn_angle=True,
                    image_solo=image_solo,
                    sate_solo=sate_solo,                    
                    action_mask=combined_action_mask,
                )
                            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
            obj_loss_cpu = losses["obj_loss"].item()
            smooth_loss_cpu = losses["smooth_loss"].item()     
                                      
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            wandb.log({"obj_loss": obj_loss_cpu})
            wandb.log({"smooth_loss": smooth_loss_cpu})    
                        
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)
                    #numbered_path = os.path.join(project_folder, f"ema_latest.pth")

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                losses['obj_loss'] = obj_loss_cpu
                losses['smooth_loss'] = smooth_loss_cpu                  
                 
                for icl in range(9):
                    losses['action_loss_' + str(icl)] = loss_list[icl]['action_loss'].item()    
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    #print(key)
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation_map2(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_viz_cur_map,
                    combined_viz_goal_map,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    mask_number,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
                visualize_lelan_eval(
                    batch_viz_obs_images_lan,
                    batch_viz_goal_images_lan,
                    goal_image_full_lan,
                    goal_image_full_8_lan,                      
                    goal_pos_lan,
                    obj_inst_lan,
                    metric_waypoint_spacing*action_estfrod[B:B+Blan].detach().cpu(),
                    metric_waypoint_spacing*select_traj_ang.detach().cpu(),
                    metric_waypoint_spacing*combined_action_pred[B+Bsub:B+Bsub+Blan].cpu(),
                    mask_number[B+Bsub:B+Bsub+Blan],
                    distance_lan,
                    project_folder,                    
                    "train",   
                    epoch,                    
                    30,                    
                    use_wandb,                                     
                    )

def train_il_exaug_dist_gnm_gps_map2_lan_ft(
    model: nn.Module,
    model_GNM: nn.Module,    
    text_encoder: nn.Module,    
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,
    noise_scheduler: DDPMScheduler,    
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    dataloader_lan: DataLoader,   
    dataloader_ft: DataLoader,          
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    #model_depth,
    #model_pedtraj,
    #device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()  

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    
    obj_loss_logger = Logger("obj_loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth_loss", "train", window_size=print_log_freq)      
    action_loss_logger_0 = Logger("action_loss_0", "train", window_size=print_log_freq)    
    action_loss_logger_1 = Logger("action_loss_1", "train", window_size=print_log_freq)    
    action_loss_logger_2 = Logger("action_loss_2", "train", window_size=print_log_freq)    
    action_loss_logger_3 = Logger("action_loss_3", "train", window_size=print_log_freq)    
    action_loss_logger_4 = Logger("action_loss_4", "train", window_size=print_log_freq)    
    action_loss_logger_5 = Logger("action_loss_5", "train", window_size=print_log_freq)    
    action_loss_logger_6 = Logger("action_loss_6", "train", window_size=print_log_freq)    
    action_loss_logger_7 = Logger("action_loss_7", "train", window_size=print_log_freq)    
    action_loss_logger_8 = Logger("action_loss_8", "train", window_size=print_log_freq)         
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger, 
        "obj_loss": obj_loss_logger,
        "smooth_loss": smooth_loss_logger,          
        "action_loss_0": action_loss_logger_0, 
        "action_loss_1": action_loss_logger_1, 
        "action_loss_2": action_loss_logger_2, 
        "action_loss_3": action_loss_logger_3, 
        "action_loss_4": action_loss_logger_4, 
        "action_loss_5": action_loss_logger_5, 
        "action_loss_6": action_loss_logger_6,     
        "action_loss_7": action_loss_logger_7, 
        "action_loss_8": action_loss_logger_8,                    
    }
    
    #D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    #K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))
    #
    #mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    #mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    #mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    #           
    dataloader_sub_iter = iter(dataloader_sub)           
    dataloader_lan_iter = iter(dataloader_lan)   
    dataloader_ft_iter = iter(dataloader_ft) 
            
    #map_image_gen = MapTileCache("/home/noriaki/Documents/map_cache/map_tiles_satellite")
    map_image_gen = MapTileCache(path_mapcache + "/map_tiles_satellite")
    transform_PIL_tensor = transforms.ToTensor()
    #ema_model.eval() 
    text_encoder.eval().to(device)
    
    model_GNM.eval().to(device)
    #model2 = copy.deepcopy(model).to(device2)            
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                obs_image_crop,                 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                current_map_image,
                goal_map_image,
                ped_list_no_trans,
                robot_list,
                #lat,
                #lon,
                #compass,
                #lat_cur,
                #lon_cur,
                #compass_cur,                
            ) = data
            
            #gc.collect()
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)    
            try:
                (
                    obs_images_lan, 
                    goal_image_lan,
                    cur_large_img_lan,
                    goal_pos_lan,
                    obj_inst_lan,
                    goal_pos_norm_lan,    
                    goal_image_full_lan,
                    goal_image_full_8_lan,                    
                    distance_lan,            
                    action_mask_lan,
                ) = next(dataloader_lan_iter)
            except StopIteration:
                dataloader_lan_iter = iter(dataloader_lan) 
                (
                    obs_images_lan, 
                    goal_image_lan,
                    cur_large_img_lan,
                    goal_pos_lan,
                    obj_inst_lan,
                    goal_pos_norm_lan,     
                    goal_image_full_lan,
                    goal_image_full_8_lan,                    
                    distance_lan,  
                    action_mask_lan,                                                   
                ) = next(dataloader_lan_iter)  
            try:
                (
                    obs_images_ft, 
                    obs_images_crop_ft,  
                    cur_large_img_ft,
                    goal_pos_ft,
                    goal_image_full_ft,
                    goal_image_full_8_ft,                    
                    distance_ft,  
                    cur_map_ft,
                    goal_map_ft,          
                    action_mask_ft,
                ) = next(dataloader_ft_iter)
            except StopIteration:
                dataloader_ft_iter = iter(dataloader_ft) 
                (
                    obs_images_ft, 
                    obs_images_crop_ft,  
                    cur_large_img_ft,
                    goal_pos_ft,
                    goal_image_full_ft,
                    goal_image_full_8_ft,                    
                    distance_ft,  
                    cur_map_ft,
                    goal_map_ft,          
                    action_mask_ft,                                             
                ) = next(dataloader_ft_iter)  

            Bft, _, H, W = obs_images_ft.size()             
            #current_map_image_ft = torch.zeros(Bft, 3, 96, 96)
            #goal_map_image_ft = torch.zeros(Blan, 3, 96, 96)            
            goal_image2_ft = transform(goal_image_full_8_ft).to(device)
             
            obs_images_ft_list = torch.split(obs_images_ft, 3, dim=1)
            obs_images_ft = [transform(obs_image_ft).to(device) for obs_image_ft in obs_images_ft_list]
            #obs_image_ft_map = obs_images_ft[-1]                                                     

            obs_image_ft = torch.cat(obs_images_ft, dim=1)
            
            obs_images_crop_ft_list = torch.split(obs_images_crop_ft, 3, dim=1)
            obs_images_crop_ft = [transform(obs_image_ft).to(device) for obs_image_ft in obs_images_crop_ft_list]
            obs_image_ft_map = obs_images_crop_ft[-1]                                                     

            obs_image_crop_ft = torch.cat(obs_images_crop_ft, dim=1)            
            
            map_images_ft = torch.cat((transform(cur_map_ft).to(device), transform(goal_map_ft).to(device), obs_image_ft_map), axis=1)  
            
            goal_image_ft = transform(goal_image_full_ft).to(device)
            distance_ft = distance_ft.float().to(device) 
            action_mask_ft = action_mask_ft.to(device)
            goal_pose_gps_ft = goal_pos_ft
            feat_text_ft = torch.zeros(Bft, 512).to(device) 
            cur_large_img_ft = transform(cur_large_img_ft).to(device)
                                                               
            Blan, _, H, W = obs_images_lan.size()             
            current_map_image_lan = torch.zeros(Blan, 3, 96, 96)
            goal_map_image_lan = torch.zeros(Blan, 3, 96, 96)            
            goal_images_crop = transform(goal_image_lan).to(device)
            goal_image2_lan = transform(goal_image_full_8_lan).to(device) 
            
            obs_images_lan_list = torch.split(obs_images_lan, 3, dim=1)
            #curobs_image_lan = obs_images_lan_list[-1]              
            
            batch_viz_obs_images_lan = TF.resize((255.0*obs_images_lan).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images_lan = TF.resize((255.0*goal_image_lan).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])

            obs_images_lan = [transform(obs_image_lan).to(device) for obs_image_lan in obs_images_lan_list]
            obs_image_lan_map = obs_images_lan_list[-1]  
            
            obs_image_lan = torch.cat(obs_images_lan, dim=1)
            obs_image_lan_nomad = torch.cat(obs_images_lan[-2:], dim=1)
            goal_pos_norm_lan = goal_pos_norm_lan.to(device) 
                            
            cur_large_img_lan = transform(cur_large_img_lan).to(device)
            #, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)
            
            dis_obj = torch.sqrt(goal_pos_lan[:,1:2]**2 + goal_pos_lan[:,0:1]**2)
            #print(dis_obj.size(), (goal_pos_lan[:,1:2]/dis_obj).size())
            goal_pose_gps_lan = torch.cat((goal_pos_lan[:,1:2], -goal_pos_lan[:,0:1], goal_pos_lan[:,1:2]/dis_obj, -goal_pos_lan[:,0:1]/dis_obj), axis=1)
            #print(goal_pos_lan.size(), goal_pose_gps_lan.size())
            map_images_lan = torch.cat((transform(current_map_image_lan).to(device), transform(goal_map_image_lan).to(device), obs_image_lan_map.to(device)), axis=1)
            goal_image_lan = transform(goal_image_full_lan).to(device)
            distance_lan = distance_lan.float().to(device) 
            action_mask_lan = action_mask_lan.to(device)
                        
            batch_obj_inst_lan = clip.tokenize(obj_inst_lan, truncate=True).to(device) 
            with torch.no_grad():  
                feat_text_lan = text_encoder.encode_text(batch_obj_inst_lan)            
                
            #print("feat_text", feat_text_lan.size(), feat_text_lan.dtype)
            #print("batch_obs_images_lan", cur_large_img_lan.size())  
            
            with torch.no_grad():
                select_traj = supervision_from_nomad(
                    ema_model_nomad,
                    noise_scheduler,
                    obs_image_lan_nomad,
                    goal_images_crop,
                    batch_viz_obs_images_lan,
                    batch_viz_goal_images_lan,
                    goal_pos_norm_lan,
                    device,
                    project_folder,
                    epoch,
                    Blan,
                    i,                
                    30,
                    use_wandb,
                    )               
            
            """
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item()).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))
            """
            """
            lat_dummy = 37.87370638591221
            lon_dummy =  -122.26739537451519
            compass_dummy = 0.0
            
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_dummy, lon_dummy, compass_dummy).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_ViNTLeRobotDataset_IL2_gps_map_cropposition(lat_dummy, lon_dummy, compass_dummy, 50.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))            
            """
            """
            #print("before", lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item())
            #print("before", lat_dummy, lon_dummy, compass_dummy)
            #print("after", new_lat_1, new_lon_1, new_heading_1)
            #print("delta", new_lat_1-lat_cur[0].item(), new_lon_1-lon_cur[0].item())
            #print("delta", new_lat_1-lat_dummy, new_lon_1-lon_dummy)            
            #current_map_image[1] = current_map_image[0]
            current_map_image[1] = cur_map_gen1
            current_map_image[2] = current_map_image[0]
            current_map_image[3] = current_map_image[0]
            #goal_map_image[1] = goal_map_image[0]       

            new_lat_2, new_lon_2, new_heading_2 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 10.0, -30.0/180*3.1415)
            goal_map_gen2 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_2, new_lon_2, new_heading_2).resize((96,96), resample=Image.Resampling.LANCZOS))
            new_lat_3, new_lon_3, new_heading_3 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, -10.0, +30.0/180*3.1415)
            goal_map_gen3 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_3, new_lon_3, new_heading_3).resize((96,96), resample=Image.Resampling.LANCZOS))
                                         
            goal_map_image[1] = goal_map_gen1               
            goal_map_image[2] = goal_map_gen2  
            goal_map_image[3] = goal_map_gen3                                           
            """
            Bsub, _, H, W = obs_image_sub.size()  
            current_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            goal_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            cur_large_img_sub = torch.zeros(Bsub, 3, 224, 224).to(device)  
            feat_text_sub = torch.zeros(Bsub, 512).to(device)  
             
            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub_map = obs_images_sub[-1].to(device)
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            map_images_sub = torch.cat((transform(current_map_image_sub).to(device), transform(goal_map_image_sub).to(device), obs_image_sub_map), axis=1)
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)
            #model_outputs = model(obs_image, goal_image)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
            #print("goal_pose_gps_sub", goal_pose_gps_sub.size())
                        
            #print("data sub", obs_image_sub.size(), action_label_sub.size(), goal_mask_sub.sum())   
                    
            #if psutil.virtual_memory().percent > 90.0:
            #    print("RAM usage (%)", psutil.virtual_memory().percent)
            #    break
                
            """
            current_image_depth = (current_image.to(device2))*mask_360_torch
            B, _, H, W = current_image_depth.size()    
            
            with torch.no_grad():
                #depth estimation
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device)   
            """
            
            B, _, H, W = obs_image.size()  
            
            #print("batch size", B, Bsub)
            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images_crop = torch.split(obs_image_crop, 3, dim=1)            
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            cur_large_img = torch.zeros(B, 3, 224, 224).to(device)  
            feat_text = torch.zeros(B, 512).to(device)  
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            #obs_image_map = obs_images[-1].to(device)
            obs_images_crop = [transform(obs_image).to(device) for obs_image in obs_images_crop]
            obs_image_map_crop = obs_images_crop[-1].to(device)                                      
                                             
            #obs_image_map_crop[1] = obs_image_map_crop[0]
            #obs_image_map_crop[2] = obs_image_map_crop[0]
            #obs_image_map_crop[3] = obs_image_map_crop[0]                                
                                                                          
            map_images = torch.cat((transform(current_map_image).to(device), transform(goal_map_image).to(device), obs_image_map_crop), axis=1)            
            
            obs_image = torch.cat(obs_images, dim=1)
            obs_image_crop = torch.cat(obs_images_crop, dim=1)            
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            #obs_image_crop[1] = obs_image_crop[0]
            #obs_image_crop[2] = obs_image_crop[0]
            #obs_image_crop[3] = obs_image_crop[0]      

            batch_goal_pos = goal_pos.to(device)
        
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
            #print("goal_pose_gps", goal_pose_gps.size())
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        
            #print(goal_mask)

            """
            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]
                    #print(ig, igr)
            """
            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            #print(obs_image.size(), obs_image_sub.size())
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_obs_image_crop = torch.cat((obs_image_crop, obs_image_sub, obs_image_lan, obs_image_crop_ft), axis=0)
            combined_goal_image_crop = torch.cat((goal_image, goal_image_sub, goal_image_lan, goal_image_ft), axis=0)            
            combined_cur_large_img = torch.cat((cur_large_img, cur_large_img_sub, cur_large_img_lan, cur_large_img_ft), axis=0)    
            combined_feat_text = torch.cat((feat_text, feat_text_sub, feat_text_lan, feat_text_ft), axis=0)  
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
            """
            print("obs_image_ft", obs_image_ft.max(), obs_image_ft.min())
            print("goal_image_ft", goal_image_ft.max(), goal_image_ft.min())
            print("cur_large_img_ft", cur_large_img_ft.max(), cur_large_img_ft.min())
            print("feat_text_ft", feat_text_ft.max(), feat_text_ft.min())
            print("map_images_ft", map_images_ft.max(), map_images_ft.min())
            """                         
            #print(distance.mean(), dist_label_sub.mean())      
            #print("distance", distance)
            #print("dist_label_sub", dist_label_sub)model2   
            #print(distance.max(), distance.min(), dist_label_sub.max(), dist_label_sub.min())
            combined_distance = torch.clip(torch.cat((distance, dist_label_sub, distance_lan, distance_ft), axis=0), min=0.0, max=20.0) 
            #combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub, action_mask_lan, action_mask_ft), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            
            combined_viz_cur_map = torch.cat((current_map_image, current_map_image_sub), axis=0)
            combined_viz_goal_map = torch.cat((goal_map_image, goal_map_image_sub), axis=0)            
            
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub, goal_pose_gps_lan, goal_pose_gps_ft), axis=0).to(device) 
            
            combined_map_images = torch.cat((map_images, map_images_sub, map_images_lan, map_images_ft), axis=0)
            #print(local_yaw.size())
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)     
            
            goal_mask_frod = []
            goal_mask_gnm = []
            goal_mask_lan = []     
            goal_mask_ft = []                   
            for idf in range(B):
                if distance[idf] <= 20:
                    goal_mask_frod.append(random.randint(0,6))
                    #goal_mask_frod.append(0)
                else:
                    goal_mask_frod.append(random.randint(0,5))
                    #goal_mask_frod.append(0)
                    
            for idg in range(Bsub):
                if dist_label_sub[idg] <= 20:
                    goal_mask_gnm.append(random.randint(4,6))
                else:
                    goal_mask_gnm.append(random.randint(4,5))
                    
            my_list = [4,7,8]
            for idl in range(Blan):
                if distance_lan[idl] == 0:
                    if random.random() > 0.5:
                        goal_mask_lan.append(random.choice(my_list))   
                    else:
                        goal_mask_lan.append(8)   
                else:
                    goal_mask_lan.append(random.randint(6,7))                

            for idt in range(Bft):
                if distance_ft[idt] <= 20:
                    goal_mask_ft.append(random.randint(0,6))
                    #goal_mask_frod.append(0)
                else:
                    goal_mask_ft.append(random.randint(0,5))
                    #goal_mask_frod.append(0)
                                                        
            """                                            
            goal_mask_f = torch.randint(0,3,(B,)).to(device) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            #goal_mask_sub = torch.zeros(Bsub).to(device) #GNM: Goal pose only
            goal_mask_sub = torch.ones(Bsub).to(device) #GNM: Goal pose only            
            goal_mask_select = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            """
            goal_mask_select = torch.tensor(goal_mask_frod + goal_mask_gnm + goal_mask_lan + goal_mask_ft).to(device)            
            combined_action_pred, combined_dist_pred, mask_number = model(combined_obs_image_crop, combined_goal_pos_gps, combined_map_images, combined_goal_image_crop, goal_mask_select, combined_feat_text, combined_cur_large_img)   
            #print(combined_action_pred.mean(), combined_dist_pred.mean(), combined_distance.mean(), combined_dist_pred.size(), combined_distance.size())
            
            #print(combined_dist_pred.mean(), combined_dist_pred.max(), combined_dist_pred.min())
            #print(combined_distance.mean(), combined_distance.max(), combined_distance.min())
            
            #print("pred", combined_dist_pred)
            #print("gt", combined_distance)            
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            with torch.no_grad():
                dist_estfrod, action_estfrod = model_GNM(obs_image, goal_image)   
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image, combined_goal_image)  
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image.to(device2), combined_goal_image.to(device2))  
                #dist_estfrod, combined_actions = model(combined_obs_image, combined_goal_image)                  
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
            #with torch.no_grad():
            #    dist_estfrod, combined_actions = model2(combined_obs_image.to(device2), combined_goal_image.to(device2))   
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            rsize = 0.3*torch.ones(B + Blan + Bft, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B + Blan + Bft, 1, 1).to(device)   
            linear_vel_old = 0.5*torch.ones(B + Blan + Bft, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B + Blan + Bft, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
            
            obs_image_mbra = torch.cat((obs_image, obs_image_lan, obs_image_ft), axis=0)
            goal_image_mbra = torch.cat((goal_image2, goal_image2_lan, goal_image2_ft), axis=0)
             
            with torch.no_grad():
                #linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image_mbra, goal_image_mbra, rsize, delay, vel_past)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
            
            
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            #print(x_traj_cat.size(), z_traj_cat.size(), yaw_traj_cat.size())
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            
            mask_0 = (distance_lan == 0).float().unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            mask_non0 = (distance_lan != 0).float().unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            
            #print(select_traj.size(), action_estfrod[B:B+Blan].size())    
            select_traj_x1 = torch.cat((torch.zeros(Blan, 1, 1).to(device), select_traj[:,0:7,0:1]), axis=1)
            select_traj_x2 = select_traj[:,:,0:1]
            select_traj_y1 = torch.cat((torch.zeros(Blan, 1, 1).to(device), select_traj[:,0:7,1:2]), axis=1)
            select_traj_y2 = select_traj[:,:,1:2]          
            dist_select_traj = torch.sqrt((select_traj_x2 - select_traj_x1)**2 + (select_traj_y2 - select_traj_y1)**2)
            cos_select_traj = (select_traj_x2 - select_traj_x1)/dist_select_traj
            sin_select_traj = (select_traj_y2 - select_traj_y1)/dist_select_traj
            select_traj_ang = torch.cat((select_traj, cos_select_traj, sin_select_traj), axis=2)
            
            action_label_lan = mask_non0*action_estfrod[B:B+Blan].detach().to(device) + mask_0*select_traj_ang.detach().to(device)
            action_label_ft = action_estfrod[B+Blan:B+Blan+Bft].detach().to(device)
            #print(action_label_lan)
            combined_actions = torch.cat((action_estfrod[0:B].detach().to(device), action_label_sub, action_label_lan, action_label_ft), axis=0)   
            
            tar_obj_pose = (goal_pos_lan.to(device))/metric_waypoint_spacing
            action_pred_lan = combined_action_pred[B+Bsub:B+Bsub+Blan]
            mask_lan = (distance_lan == 0)*((goal_mask_select == 7) + (goal_mask_select == 8))[B+Bsub:B+Bsub+Blan]
            #print(B, Blan, action_estfrod[0:B].size(), action_label_sub.size(), action_estfrod[B:Blan].size())
            """                                   
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )
            """
            #print(goal_mask_select.size(), combined_distance.size(), combined_actions.size())
            loss_list = []
            for icl in range(9):
                mask_task = goal_mask_select == icl
                #print(mask_task)
                losses = _compute_losses(
                    dist_label=combined_distance[mask_task].to(device),
                    action_label=combined_actions[mask_task].to(device),
                    dist_pred=combined_dist_pred[mask_task],
                    action_pred=combined_action_pred[mask_task],
                    alpha=0.5,
                    learn_angle=True,
                    action_mask=combined_action_mask[mask_task],
                )   
                loss_list.append(losses)
                action_loss_cpu = losses["action_loss"].item()                 
                wandb.log({"action_loss_" + str(icl): action_loss_cpu}) 
                        
            #losses = _compute_losses(
            #    dist_label=combined_distance.to(device),
            #    action_label=combined_actions.to(device),
            #    dist_pred=combined_dist_pred,
            #    action_pred=combined_action_pred,
            #    alpha=0.5,
            #    learn_angle=True,
            #    action_mask=combined_action_mask,
            #)
            losses = _compute_losses_lan(
                dist_label=combined_distance.to(device),
                action_label=combined_actions.to(device),
                dist_pred=combined_dist_pred,
                action_pred=combined_action_pred,
                pose_obj_label=tar_obj_pose[mask_lan],
                pose_obj_pred=action_pred_lan[:,-1,0:2][mask_lan],
                alpha=0.5,
                learn_angle=True,    
                image_solo=False,
                sate_solo=False,                               
                action_mask=combined_action_mask,
            )
                                        
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
            obj_loss_cpu = losses["obj_loss"].item()
            smooth_loss_cpu = losses["smooth_loss"].item()    
                                      
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            wandb.log({"obj_loss": obj_loss_cpu})
            wandb.log({"smooth_loss": smooth_loss_cpu}) 
                        
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)
                    #numbered_path = os.path.join(project_folder, f"ema_latest.pth")

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                losses['obj_loss'] = obj_loss_cpu
                losses['smooth_loss'] = smooth_loss_cpu                   
                for icl in range(9):
                    losses['action_loss_' + str(icl)] = loss_list[icl]['action_loss'].item()    
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    #print(key)
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation_map2(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_viz_cur_map,
                    combined_viz_goal_map,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    mask_number,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
                visualize_lelan_eval(
                    batch_viz_obs_images_lan,
                    batch_viz_goal_images_lan,
                    goal_image_full_lan,
                    goal_image_full_8_lan,                      
                    goal_pos_lan,
                    obj_inst_lan,
                    metric_waypoint_spacing*action_estfrod[B:B+Blan].detach().cpu(),
                    metric_waypoint_spacing*select_traj_ang.detach().cpu(),
                    combined_action_pred[B+Bsub:B+Bsub+Blan].cpu(),
                    mask_number[B+Bsub:B+Bsub+Blan],
                    distance_lan,
                    project_folder,                    
                    "train",   
                    epoch,                    
                    30,                    
                    use_wandb,                                     
                    )

def train_il_exaug_dist_gnm_gps_map2_lan_bdd(
    model: nn.Module,
    model_GNM: nn.Module,
    model_bdd: nn.Module,
    text_encoder: nn.Module,    
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,
    noise_scheduler: DDPMScheduler,    
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    latest_path: str,
    dataloader: DataLoader,
    dataloader_sub: DataLoader,    
    dataloader_lan: DataLoader,   
    dataloader_bdd: DataLoader,          
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    no_emamodel: bool,
    #model_depth,
    #model_pedtraj,
    #device2,      
    len_traj_pred: int,       
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()  

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)    
    obj_loss_logger = Logger("obj_loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth_loss", "train", window_size=print_log_freq)      
    action_loss_logger_0 = Logger("action_loss_0", "train", window_size=print_log_freq)    
    action_loss_logger_1 = Logger("action_loss_1", "train", window_size=print_log_freq)    
    action_loss_logger_2 = Logger("action_loss_2", "train", window_size=print_log_freq)    
    action_loss_logger_3 = Logger("action_loss_3", "train", window_size=print_log_freq)    
    action_loss_logger_4 = Logger("action_loss_4", "train", window_size=print_log_freq)    
    action_loss_logger_5 = Logger("action_loss_5", "train", window_size=print_log_freq)    
    action_loss_logger_6 = Logger("action_loss_6", "train", window_size=print_log_freq)    
    action_loss_logger_7 = Logger("action_loss_7", "train", window_size=print_log_freq)    
    action_loss_logger_8 = Logger("action_loss_8", "train", window_size=print_log_freq)         
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger, 
        "obj_loss": obj_loss_logger,
        "smooth_loss": smooth_loss_logger,          
        "action_loss_0": action_loss_logger_0, 
        "action_loss_1": action_loss_logger_1, 
        "action_loss_2": action_loss_logger_2, 
        "action_loss_3": action_loss_logger_3, 
        "action_loss_4": action_loss_logger_4, 
        "action_loss_5": action_loss_logger_5, 
        "action_loss_6": action_loss_logger_6,     
        "action_loss_7": action_loss_logger_7, 
        "action_loss_8": action_loss_logger_8,                    
    }
    
    #D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    #K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))
    #
    #mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    #mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    #mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    #           
    dataloader_sub_iter = iter(dataloader_sub)           
    dataloader_lan_iter = iter(dataloader_lan)   
    dataloader_bdd_iter = iter(dataloader_bdd) 
            
    #map_image_gen = MapTileCache("/media/noriaki/Noriaki_Data2/map_cache/map_tiles_satellite")
    map_image_gen = MapTileCache(path_mapcache + "/map_tiles_satellite")
    transform_PIL_tensor = transforms.ToTensor()
    #ema_model.eval() 
    text_encoder.eval().to(device)
    
    model_GNM.eval().to(device)
    model_bdd.eval().to(device)    
    #model2 = copy.deepcopy(model).to(device2)            
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                obs_image_crop,                 
                goal_image,
                goal_image2,                
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,              
                actions_raw,
                obs_image_future,
                #_, 
                id_num,
                action_mask,
                current_map_image,
                goal_map_image,
                ped_list_no_trans,
                robot_list,
                #lat,
                #lon,
                #compass,
                #lat_cur,
                #lon_cur,
                #compass_cur,                
            ) = data
            
            #gc.collect()
            try:
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                    
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)
            except StopIteration:
                dataloader_sub_iter = iter(dataloader_sub) 
                (
                    obs_image_sub,
                    goal_image_sub,
                    action_label_sub,
                    dist_label_sub,
                    goal_pos_sub,
                    goal_yaw_sub,                     
                    dataset_index_sub,
                    action_mask_sub,
                    #_,
                    #_,
                ) = next(dataloader_sub_iter)    
            try:
                (
                    obs_images_lan, 
                    goal_image_lan,
                    cur_large_img_lan,
                    goal_pos_lan,
                    obj_inst_lan,
                    goal_pos_norm_lan,    
                    goal_image_full_lan,
                    goal_image_full_8_lan,                    
                    distance_lan,            
                    action_mask_lan,
                ) = next(dataloader_lan_iter)
            except StopIteration:
                dataloader_lan_iter = iter(dataloader_lan) 
                (
                    obs_images_lan, 
                    goal_image_lan,
                    cur_large_img_lan,
                    goal_pos_lan,
                    obj_inst_lan,
                    goal_pos_norm_lan,     
                    goal_image_full_lan,
                    goal_image_full_8_lan,                    
                    distance_lan,  
                    action_mask_lan,                                                   
                ) = next(dataloader_lan_iter)  
            try:
                (
                    obs_images_bdd, 
                    obs_images_crop_bdd,  
                    cur_large_img_bdd,
                    goal_pos_bdd,
                    goal_image_full_bdd,
                    goal_image_full_8_bdd,                    
                    distance_bdd,   
                    action_mask_bdd,
                ) = next(dataloader_bdd_iter)
            except StopIteration:
                dataloader_bdd_iter = iter(dataloader_bdd) 
                (
                    obs_images_bdd, 
                    obs_images_crop_bdd,  
                    cur_large_img_bdd,
                    goal_pos_bdd,
                    goal_image_full_bdd,
                    goal_image_full_8_bdd,                    
                    distance_bdd,      
                    action_mask_bdd,                                             
                ) = next(dataloader_bdd_iter)     

            Bbdd, _, H, W = obs_images_bdd.size()                       
            batch_viz_obs_images_bdd = TF.resize((255.0*obs_images_bdd).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images_bdd = TF.resize((255.0*goal_image_full_bdd).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])             
            batch_viz_goal_images_bdd_8 = TF.resize((255.0*goal_image_full_8_bdd).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])            
            
            goal_image2_bdd = transform(goal_image_full_8_bdd).to(device)
             
            obs_images_bdd_list = torch.split(obs_images_bdd, 3, dim=1)
            obs_images_bdd = [transform(obs_image_bdd).to(device) for obs_image_bdd in obs_images_bdd_list]                                                    

            obs_image_bdd = torch.cat(obs_images_bdd, dim=1)
            
            obs_images_crop_bdd_list = torch.split(obs_images_crop_bdd, 3, dim=1)
            obs_images_crop_bdd = [transform(obs_image_bdd).to(device) for obs_image_bdd in obs_images_crop_bdd_list]
            obs_image_bdd_map = obs_images_crop_bdd[-1]                                                     

            obs_image_crop_bdd = torch.cat(obs_images_crop_bdd, dim=1)
                        
            current_map_image_bdd = torch.zeros(Bbdd, 3, 96, 96)
            goal_map_image_bdd = torch.zeros(Bbdd, 3, 96, 96)                              
            map_images_bdd = torch.cat((transform(current_map_image_bdd).to(device), transform(goal_map_image_bdd).to(device), obs_image_bdd_map), axis=1)  
            
            goal_image_bdd = transform(goal_image_full_bdd).to(device)
            distance_bdd = distance_bdd.float().to(device) 
            action_mask_bdd = action_mask_bdd.to(device)
            goal_pose_gps_bdd = goal_pos_bdd
            feat_text_bdd = torch.zeros(Bbdd, 512).to(device) 
            cur_large_img_bdd = transform(cur_large_img_bdd).to(device)
                                                               
            Blan, _, H, W = obs_images_lan.size()             
            current_map_image_lan = torch.zeros(Blan, 3, 96, 96)
            goal_map_image_lan = torch.zeros(Blan, 3, 96, 96)            
            goal_images_crop = transform(goal_image_lan).to(device)
            goal_image2_lan = transform(goal_image_full_8_lan).to(device) 
            
            obs_images_lan_list = torch.split(obs_images_lan, 3, dim=1)
            #curobs_image_lan = obs_images_lan_list[-1]              
            
            batch_viz_obs_images_lan = TF.resize((255.0*obs_images_lan).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images_lan = TF.resize((255.0*goal_image_lan).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])

            obs_images_lan = [transform(obs_image_lan).to(device) for obs_image_lan in obs_images_lan_list]
            obs_image_lan_map = obs_images_lan_list[-1]  
            
            obs_image_lan = torch.cat(obs_images_lan, dim=1)
            obs_image_lan_nomad = torch.cat(obs_images_lan[-2:], dim=1)
            goal_pos_norm_lan = goal_pos_norm_lan.to(device) 
                            
            cur_large_img_lan = transform(cur_large_img_lan).to(device)
            #, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)
            
            dis_obj = torch.sqrt(goal_pos_lan[:,1:2]**2 + goal_pos_lan[:,0:1]**2)
            #print(dis_obj.size(), (goal_pos_lan[:,1:2]/dis_obj).size())
            goal_pose_gps_lan = torch.cat((goal_pos_lan[:,1:2], -goal_pos_lan[:,0:1], goal_pos_lan[:,1:2]/dis_obj, -goal_pos_lan[:,0:1]/dis_obj), axis=1)
            #print(goal_pos_lan.size(), goal_pose_gps_lan.size())
            map_images_lan = torch.cat((transform(current_map_image_lan).to(device), transform(goal_map_image_lan).to(device), obs_image_lan_map.to(device)), axis=1)
            goal_image_lan = transform(goal_image_full_lan).to(device)
            distance_lan = distance_lan.float().to(device) 
            action_mask_lan = action_mask_lan.to(device)
                        
            batch_obj_inst_lan = clip.tokenize(obj_inst_lan, truncate=True).to(device) 
            with torch.no_grad():  
                feat_text_lan = text_encoder.encode_text(batch_obj_inst_lan)            
                
            #print("feat_text", feat_text_lan.size(), feat_text_lan.dtype)
            #print("batch_obs_images_lan", cur_large_img_lan.size())  
            
            with torch.no_grad():
                select_traj = supervision_from_nomad(
                    ema_model_nomad,
                    noise_scheduler,
                    obs_image_lan_nomad,
                    goal_images_crop,
                    batch_viz_obs_images_lan,
                    batch_viz_goal_images_lan,
                    goal_pos_norm_lan,
                    device,
                    project_folder,
                    epoch,
                    Blan,
                    i,                
                    30,
                    use_wandb,
                    )               
            
            """
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item()).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))
            """
            """
            lat_dummy = 37.87370638591221
            lon_dummy =  -122.26739537451519
            compass_dummy = 0.0
            
            cur_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(lat_dummy, lon_dummy, compass_dummy).resize((96,96), resample=Image.Resampling.LANCZOS))    
            
            new_lat_1, new_lon_1, new_heading_1 = transform_ViNTLeRobotDataset_IL2_gps_map_cropposition(lat_dummy, lon_dummy, compass_dummy, 50.0, 0.0, 0.0)        
            goal_map_gen1 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_1, new_lon_1, new_heading_1).resize((96,96), resample=Image.Resampling.LANCZOS))            
            """
            """
            #print("before", lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item())
            #print("before", lat_dummy, lon_dummy, compass_dummy)
            #print("after", new_lat_1, new_lon_1, new_heading_1)
            #print("delta", new_lat_1-lat_cur[0].item(), new_lon_1-lon_cur[0].item())
            #print("delta", new_lat_1-lat_dummy, new_lon_1-lon_dummy)            
            #current_map_image[1] = current_map_image[0]
            current_map_image[1] = cur_map_gen1
            current_map_image[2] = current_map_image[0]
            current_map_image[3] = current_map_image[0]
            #goal_map_image[1] = goal_map_image[0]       

            new_lat_2, new_lon_2, new_heading_2 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, 10.0, -30.0/180*3.1415)
            goal_map_gen2 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_2, new_lon_2, new_heading_2).resize((96,96), resample=Image.Resampling.LANCZOS))
            new_lat_3, new_lon_3, new_heading_3 = transform_position(lat_cur[0].item(), lon_cur[0].item(), compass_cur[0].item(), 30.0, -10.0, +30.0/180*3.1415)
            goal_map_gen3 = transform_PIL_tensor(map_image_gen.get_map_view(new_lat_3, new_lon_3, new_heading_3).resize((96,96), resample=Image.Resampling.LANCZOS))
                                         
            goal_map_image[1] = goal_map_gen1               
            goal_map_image[2] = goal_map_gen2  
            goal_map_image[3] = goal_map_gen3                                           
            """
            Bsub, _, H, W = obs_image_sub.size()  
            current_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            goal_map_image_sub = torch.zeros(Bsub, 3, 96, 96)
            cur_large_img_sub = torch.zeros(Bsub, 3, 224, 224).to(device)  
            feat_text_sub = torch.zeros(Bsub, 512).to(device)  
             
            obs_images_sub = torch.split(obs_image_sub, 3, dim=1)
            viz_obs_image_sub = TF.resize(obs_images_sub[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past_sub = TF.resize(obs_images_sub[0], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images_sub = [transform(obs_image_sub).to(device) for obs_image_sub in obs_images_sub]
            obs_image_sub_map = obs_images_sub[-1].to(device)
            obs_image_sub = torch.cat(obs_images_sub, dim=1)

            map_images_sub = torch.cat((transform(current_map_image_sub).to(device), transform(goal_map_image_sub).to(device), obs_image_sub_map), axis=1)
            viz_goal_image_sub = TF.resize(goal_image_sub, VISUALIZATION_IMAGE_SIZE[::-1])
        
            goal_image_sub = transform(goal_image_sub).to(device)
            #model_outputs = model(obs_image, goal_image)

            dist_label_sub = dist_label_sub.to(device)
            action_label_sub = action_label_sub.to(device)
            action_mask_sub = action_mask_sub.to(device)
            goal_mask_sub = dist_label_sub > -1.0
            
            goal_pose_gps_sub = torch.cat((goal_pos_sub, torch.cos(goal_yaw_sub).unsqueeze(1), torch.sin(goal_yaw_sub).unsqueeze(1)), axis=1)
            #print("goal_pose_gps_sub", goal_pose_gps_sub.size())
                        
            #print("data sub", obs_image_sub.size(), action_label_sub.size(), goal_mask_sub.sum())   
                    
            #if psutil.virtual_memory().percent > 90.0:
            #    print("RAM usage (%)", psutil.virtual_memory().percent)
            #    break
                
            """
            current_image_depth = (current_image.to(device2))*mask_360_torch
            B, _, H, W = current_image_depth.size()    
            
            with torch.no_grad():
                #depth estimation
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device)   
            """
            
            B, _, H, W = obs_image.size()  
            
            #print("batch size", B, Bsub)
            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images_crop = torch.split(obs_image_crop, 3, dim=1)            
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])     
            cur_large_img = torch.zeros(B, 3, 224, 224).to(device)  
            feat_text = torch.zeros(B, 512).to(device)  
            
            obs_images_future = torch.split(obs_image_future, 3, dim=1)
                   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            #obs_image_map = obs_images[-1].to(device)
            obs_images_crop = [transform(obs_image).to(device) for obs_image in obs_images_crop]
            obs_image_map_crop = obs_images_crop[-1].to(device)                                      
                                             
            #obs_image_map_crop[1] = obs_image_map_crop[0]
            #obs_image_map_crop[2] = obs_image_map_crop[0]
            #obs_image_map_crop[3] = obs_image_map_crop[0]                                
                                                                          
            map_images = torch.cat((transform(current_map_image).to(device), transform(goal_map_image).to(device), obs_image_map_crop), axis=1)            
            
            obs_image = torch.cat(obs_images, dim=1)
            obs_image_crop = torch.cat(obs_images_crop, dim=1)            
            actions = actions.to(device)
            action_mask = action_mask.to(device)

            #obs_image_crop[1] = obs_image_crop[0]
            #obs_image_crop[2] = obs_image_crop[0]
            #obs_image_crop[3] = obs_image_crop[0]      

            batch_goal_pos = goal_pos.to(device)
        
            goal_pose_gps = torch.cat((goal_pos, local_goal_mat[:,1,1].unsqueeze(1), local_goal_mat[:,1,0].unsqueeze(1)), axis=1)
            #print("goal_pose_gps", goal_pose_gps.size())
                
            # Get distance label
            distance = distance.float().to(device)
            goal_mask = distance > 0.1                        
            #print(goal_mask)

            """
            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]
                    #print(ig, igr)
            """
            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)         
            goal_image2 = transform(goal_image2).to(device)         
                        
            #print(obs_image.size(), obs_image_sub.size())
            combined_obs_image = torch.cat((obs_image, obs_image_sub), axis=0)
            combined_obs_image_crop = torch.cat((obs_image_crop, obs_image_sub, obs_image_lan, obs_image_crop_bdd), axis=0)
            combined_goal_image_crop = torch.cat((goal_image, goal_image_sub, goal_image_lan, goal_image_bdd), axis=0)            
            combined_cur_large_img = torch.cat((cur_large_img, cur_large_img_sub, cur_large_img_lan, cur_large_img_bdd), axis=0)    
            combined_feat_text = torch.cat((feat_text, feat_text_sub, feat_text_lan, feat_text_bdd), axis=0)  
            combined_actions_origin = torch.cat((actions, action_label_sub), axis=0)   
            """
            print("obs_image_ft", obs_image_ft.max(), obs_image_ft.min())
            print("goal_image_ft", goal_image_ft.max(), goal_image_ft.min())
            print("cur_large_img_ft", cur_large_img_ft.max(), cur_large_img_ft.min())
            print("feat_text_ft", feat_text_ft.max(), feat_text_ft.min())
            print("map_images_ft", map_images_ft.max(), map_images_ft.min())
            """                         
            #print(distance.mean(), dist_label_sub.mean())      
            #print("distance", distance)
            #print("dist_label_sub", dist_label_sub)model2   
            #print(distance.max(), distance.min(), dist_label_sub.max(), dist_label_sub.min())
            combined_distance = torch.clip(torch.cat((distance, dist_label_sub, distance_lan, distance_bdd), axis=0), min=0.0, max=20.0) 
            #combined_goal_mask = torch.cat((goal_mask, goal_mask_sub), axis=0)                   
            combined_action_mask = torch.cat((action_mask, action_mask_sub, action_mask_lan, action_mask_bdd), axis=0) 

            combined_viz_obs_image = torch.cat((viz_obs_image, viz_obs_image_sub), axis=0)                   
            combined_viz_obs_image_past = torch.cat((viz_obs_image_past, viz_obs_image_past_sub), axis=0) 
            
            combined_viz_cur_map = torch.cat((current_map_image, current_map_image_sub), axis=0)
            combined_viz_goal_map = torch.cat((goal_map_image, goal_map_image_sub), axis=0)            
            
            combined_viz_goal_image = torch.cat((viz_goal_image, viz_goal_image_sub), axis=0) 
            combined_goal_pos = torch.cat((goal_pos, goal_pos_sub), axis=0) 
            combined_goal_pos_gps = torch.cat((goal_pose_gps, goal_pose_gps_sub, goal_pose_gps_lan, goal_pose_gps_bdd), axis=0).to(device) 
            
            combined_map_images = torch.cat((map_images, map_images_sub, map_images_lan, map_images_bdd), axis=0)
            #print(local_yaw.size())
            combined_local_yaw = torch.cat((local_yaw, torch.ones(Bsub)), axis=0)     
            
            goal_mask_frod = []
            goal_mask_gnm = []
            goal_mask_lan = []     
            goal_mask_bdd = []                   
            for idf in range(B):
                if distance[idf] <= 20:
                    goal_mask_frod.append(random.randint(0,6))
                    #goal_mask_frod.append(0)
                else:
                    goal_mask_frod.append(random.randint(0,5))
                    #goal_mask_frod.append(0)
                    
            for idg in range(Bsub):
                if dist_label_sub[idg] <= 20:
                    goal_mask_gnm.append(random.randint(4,6))
                else:
                    goal_mask_gnm.append(random.randint(4,5))
                    
            my_list = [4,7,8]
            for idl in range(Blan):
                if distance_lan[idl] == 0:
                    if random.random() > 0.5:
                        goal_mask_lan.append(random.choice(my_list))   
                    else:
                        goal_mask_lan.append(8)   
                else:
                    goal_mask_lan.append(random.randint(6,7))                

            for idt in range(Bbdd):
                if distance_bdd[idt] <= 20:
                    goal_mask_bdd.append(random.randint(4,6))
                    #goal_mask_frod.append(0)
                else:
                    goal_mask_bdd.append(random.randint(4,5))
                    #goal_mask_frod.append(0)
                                                        
            """                                            
            goal_mask_f = torch.randint(0,3,(B,)).to(device) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            #goal_mask_sub = torch.zeros(Bsub).to(device) #GNM: Goal pose only
            goal_mask_sub = torch.ones(Bsub).to(device) #GNM: Goal pose only            
            goal_mask_select = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            """
            goal_mask_select = torch.tensor(goal_mask_frod + goal_mask_gnm + goal_mask_lan + goal_mask_bdd).to(device)            
            combined_action_pred, combined_dist_pred, mask_number = model(combined_obs_image_crop, combined_goal_pos_gps, combined_map_images, combined_goal_image_crop, goal_mask_select, combined_feat_text, combined_cur_large_img)   
            #print(combined_action_pred.mean(), combined_dist_pred.mean(), combined_distance.mean(), combined_dist_pred.size(), combined_distance.size())
            
            #print(combined_dist_pred.mean(), combined_dist_pred.max(), combined_dist_pred.min())
            #print(combined_distance.mean(), combined_distance.max(), combined_distance.min())
            
            #print("pred", combined_dist_pred)
            #print("gt", combined_distance)            
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            with torch.no_grad():
                dist_estfrod, action_estfrod = model_GNM(obs_image, goal_image)   
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image, combined_goal_image)  
                #dist_estfrod, combined_actions = model_GNM(combined_obs_image.to(device2), combined_goal_image.to(device2))  
                #dist_estfrod, combined_actions = model(combined_obs_image, combined_goal_image)                  
            combined_actions = torch.cat((action_estfrod.detach().to(device), action_label_sub), axis=0)   
            #with torch.no_grad():
            #    dist_estfrod, combined_actions = model2(combined_obs_image.to(device2), combined_goal_image.to(device2))   
            """
            #labeling by Robotic foundation model (IL on GNM dataset)
            rsize = 0.3*torch.ones(B + Blan, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            rsize_bdd = 0.3*torch.ones(Bbdd, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.zeros(B + Blan, 1, 1).to(device)   
            delay_bdd = torch.zeros(Bbdd, 1, 1).to(device)               
            linear_vel_old = 0.5*torch.ones(B + Blan, 6).float().to(device)
            angular_vel_old = 0.0*torch.ones(B + Blan, 6).float().to(device)
            vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
            linear_vel_old_bdd = 0.0*torch.ones(Bbdd, 6).float().to(device) #We omit the delay in BDD training (No delay in BDD dataset and we do not deplay this mode as the image-conditioned nav policy)
            angular_vel_old_bdd = 0.0*torch.ones(Bbdd, 6).float().to(device)
            vel_past_bdd = torch.cat((linear_vel_old_bdd, angular_vel_old_bdd), axis=1).unsqueeze(2)  
                        
            obs_image_mbra = torch.cat((obs_image, obs_image_lan), axis=0)
            goal_image_mbra = torch.cat((goal_image2, goal_image2_lan), axis=0)
             
            with torch.no_grad():
                #linear_vel, angular_vel, dist_estfrod = model_GNM(obs_image, goal_image2, rsize, delay, vel_past)
                linear_vel_, angular_vel_, dist_estfrod_ = model_GNM(obs_image_mbra, goal_image_mbra, rsize, delay, vel_past)
                linear_vel_bdd, angular_vel_bdd, dist_estfrod_bdd = model_bdd(obs_image_bdd, goal_image2_bdd, rsize_bdd, delay_bdd, vel_past_bdd)
            linear_vel = torch.cat((linear_vel_, linear_vel_bdd), axis=0)
            angular_vel = torch.cat((angular_vel_, angular_vel_bdd), axis=0)
            dist_estfrod = torch.cat((dist_estfrod_, dist_estfrod_bdd), axis=0)
                                                
            linear_vel_d = linear_vel
            angular_vel_d = angular_vel            
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)         
            
            
            x_traj = []
            z_traj = []
            yaw_traj = [] 
            for ic in range(len(px_ref_list)):
                x_traj.append(px_ref_list[ic].unsqueeze(1))
                z_traj.append(pz_ref_list[ic].unsqueeze(1))
                yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
            x_traj_cat = torch.cat(x_traj, axis = 1)
            z_traj_cat = torch.cat(z_traj, axis = 1)
            yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
            metric_waypoint_spacing = 0.25*0.5
            #print(x_traj_cat.size(), z_traj_cat.size(), yaw_traj_cat.size())
            action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)     
            
            mask_0 = (distance_lan == 0).float().unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            mask_non0 = (distance_lan != 0).float().unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            
            #print(select_traj.size(), action_estfrod[B:B+Blan].size())    
            select_traj_x1 = torch.cat((torch.zeros(Blan, 1, 1).to(device), select_traj[:,0:7,0:1]), axis=1)
            select_traj_x2 = select_traj[:,:,0:1]
            select_traj_y1 = torch.cat((torch.zeros(Blan, 1, 1).to(device), select_traj[:,0:7,1:2]), axis=1)
            select_traj_y2 = select_traj[:,:,1:2]          
            dist_select_traj = torch.sqrt((select_traj_x2 - select_traj_x1)**2 + (select_traj_y2 - select_traj_y1)**2)
            cos_select_traj = (select_traj_x2 - select_traj_x1)/dist_select_traj
            sin_select_traj = (select_traj_y2 - select_traj_y1)/dist_select_traj
            select_traj_ang = torch.cat((select_traj, cos_select_traj, sin_select_traj), axis=2)
            
            action_label_lan = mask_non0*action_estfrod[B:B+Blan].detach().to(device) + mask_0*select_traj_ang.detach().to(device)
            action_label_bdd = action_estfrod[B+Blan:B+Blan+Bbdd].detach().to(device)
            #print(action_label_lan)
            combined_actions = torch.cat((action_estfrod[0:B].detach().to(device), action_label_sub, action_label_lan, action_label_bdd), axis=0)   
            
            tar_obj_pose = (goal_pos_lan.to(device))/metric_waypoint_spacing
            action_pred_lan = combined_action_pred[B+Bsub:B+Bsub+Blan]
            mask_lan = (distance_lan == 0)*((goal_mask_select == 7) + (goal_mask_select == 8))[B+Bsub:B+Bsub+Blan]
            #print(B, Blan, action_estfrod[0:B].size(), action_label_sub.size(), action_estfrod[B:Blan].size())
            """                                   
            losses = _compute_losses_gps(
                action_label=combined_actions.to(device),
                action_pred=combined_action_pred,
                learn_angle=True,
                action_mask=combined_action_mask,
            )
            """
            #print(goal_mask_select.size(), combined_distance.size(), combined_actions.size())
            loss_list = []
            for icl in range(9):
                mask_task = goal_mask_select == icl
                #print(mask_task)
                losses = _compute_losses(
                    dist_label=combined_distance[mask_task].to(device),
                    action_label=combined_actions[mask_task].to(device),
                    dist_pred=combined_dist_pred[mask_task],
                    action_pred=combined_action_pred[mask_task],
                    alpha=0.5,
                    learn_angle=True,
                    action_mask=combined_action_mask[mask_task],
                )   
                loss_list.append(losses)
                action_loss_cpu = losses["action_loss"].item()                 
                wandb.log({"action_loss_" + str(icl): action_loss_cpu}) 
                        
            #losses = _compute_losses(
            #    dist_label=combined_distance.to(device),
            #    action_label=combined_actions.to(device),
            #    dist_pred=combined_dist_pred,
            #    action_pred=combined_action_pred,
            #    alpha=0.5,
            #    learn_angle=True,
            #    action_mask=combined_action_mask,
            #)
            losses = _compute_losses_lan(
                dist_label=combined_distance.to(device),
                action_label=combined_actions.to(device),
                dist_pred=combined_dist_pred,
                action_pred=combined_action_pred,
                pose_obj_label=tar_obj_pose[mask_lan],
                pose_obj_pred=action_pred_lan[:,-1,0:2][mask_lan],
                alpha=0.5,
                learn_angle=True,    
                image_solo=False,
                sate_solo=False,                               
                action_mask=combined_action_mask,
            )
            if i%3 == 1:                                        
                optimizer.zero_grad()
            losses["total_loss"].backward()            
            if i%3 == 0:
                optimizer.step()
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
            obj_loss_cpu = losses["obj_loss"].item()
            smooth_loss_cpu = losses["smooth_loss"].item()    
                                      
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})            
            wandb.log({"obj_loss": obj_loss_cpu})
            wandb.log({"smooth_loss": smooth_loss_cpu}) 
                        
            if epoch == 0 and i == 2000:
                lr_scheduler.step()
            
            if i % 5000 == 0 and i != 0:
                lr_scheduler.step()

            #if False:
            if i % 500 == 0 and i != 0:
                if no_emamodel:
                    numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
                    torch.save(ema_model.averaged_model.state_dict(), numbered_path)
                    #numbered_path = os.path.join(project_folder, f"ema_latest.pth")

                numbered_path = os.path.join(project_folder, f"{epoch}.pth")
                torch.save(model.state_dict(), numbered_path)
                torch.save(model.state_dict(), latest_path)

                # save optimizer
                numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
                latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
                torch.save(optimizer.state_dict(), latest_optimizer_path)

                # save scheduler
                numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
                latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
                torch.save(lr_scheduler.state_dict(), latest_scheduler_path)
        
            #if False:
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                losses['obj_loss'] = obj_loss_cpu
                losses['smooth_loss'] = smooth_loss_cpu                   
                for icl in range(9):
                    losses['action_loss_' + str(icl)] = loss_list[icl]['action_loss'].item()    
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    #print(key)
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:                
                visualize_il2_estimation_map2(
                    combined_viz_obs_image, 
                    combined_viz_obs_image_past,                     
                    combined_viz_goal_image,
                    combined_viz_cur_map,
                    combined_viz_goal_map,
                    combined_goal_pos,
                    combined_local_yaw,
                    combined_action_pred,
                    combined_actions,
                    combined_actions_origin,
                    mask_number,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
                visualize_lelan_eval(
                    batch_viz_obs_images_lan,
                    batch_viz_goal_images_lan,
                    goal_image_full_lan,
                    goal_image_full_8_lan,                      
                    goal_pos_lan,
                    obj_inst_lan,
                    metric_waypoint_spacing*action_estfrod[B:B+Blan].detach().cpu(),
                    metric_waypoint_spacing*select_traj_ang.detach().cpu(),
                    combined_action_pred[B+Bsub:B+Bsub+Blan].cpu(),
                    mask_number[B+Bsub:B+Bsub+Blan],
                    distance_lan,
                    project_folder,                    
                    "train",   
                    epoch,                    
                    30,                    
                    use_wandb,                                     
                    )
                visualize_BDD_eval(
                    batch_viz_obs_images_bdd,
                    batch_viz_goal_images_bdd,
                    batch_viz_goal_images_bdd_8,
                    #goal_image_full_lan,
                    #goal_image_full_8_lan,                      
                    0.5*goal_pos_bdd,
                    #obj_inst_lan,
                    metric_waypoint_spacing*action_estfrod[B+Blan:].detach().cpu(),
                    #metric_waypoint_spacing*select_traj_ang.detach().cpu(),
                    metric_waypoint_spacing*combined_action_pred[B+Bsub+Blan:].cpu(),
                    mask_number[B+Bsub+Blan:],
                    distance_bdd,
                    project_folder,                    
                    "train",   
                    epoch,                    
                    40,                    
                    use_wandb,                                     
                    )


def train_annotate(
    #model: nn.Module,
    #ema_model: EMAModel,
    #optimizer: Adam,
    #lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    #latest_path: str,
    dataloader: DataLoader,
    #transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    model_depth,
    model_pedest,
    #device2,      
    #len_traj_pred: int,       
    #alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,   
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    #model.train()
    num_batches = len(dataloader)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth_loss", "train", window_size=print_log_freq)
    geo_loss_logger = Logger("geo_loss", "train", window_size=print_log_freq)

    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "smooth_loss": smooth_loss_logger,
        "geo_loss": geo_loss_logger,
    }
    
    #D = np.array(np.load('/mnt/ephemeral2/Learning-to-Drive-Anywhere-via-MBRA/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    #K = np.array(np.load('/mnt/ephemeral2/Learning-to-Drive-Anywhere-via-MBRA/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))

    mask_360 = np.loadtxt(open("/mnt/ephemeral2/Learning-to-Drive-Anywhere-via-MBRA/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device)
    #print(mask_360_torch.size())

    """
    mask_gs = np.zeros((1, 128, 416), dtype = 'float32')
    mask_recon = np.zeros((1, 128, 416), dtype = 'float32')
    
    for i in range(416):
        for j in range(128):
            if ((i - center_w)**2)/(0.5*416)**2 + ((j - center_h)**2)/(0.5*128)**2 <= 1:
                mask_gs[0,j,i] = 1.0
            if ((i - center_w)**2)/(0.5*416*0.95)**2 + ((j - center_h)**2)/(0.5*128*1.5)**2 <= 1:
                mask_recon[0,j,i] = 1.0                
    """           
    #hfov, vfov = calculate_fov(K, (1024, 576))
    #print("D", D)
    #print("K", K)                
    transform_3d = T.Resize(size = (96,96))   
    yaw = []     
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                id_num,
                filtered_position,     
                filtered_heading,         
                utm_position,        
                compass_heading,                           
                gyroscope,      
                accelerometer,         
                wheel_rpm,       
                action,                            
                action_original,       
                episode_index,              
                frame_index,       
                timestamp, 
                utm_zone_letter,
                utm_zone_number,
                #current_image,
                #image_raw,
                lat,
                lon,
                #video_id_number_1,
                #video_id_number_2,  
                relative_mat,              
            ) = data      
            """
            current_image_depth = (current_image.to(device))*mask_360_torch
            current_image_raw = image_raw.to(device)
            
            current_image_depth_flip = torch.flip(current_image_depth, dims=[3]) 
            current_image_depth_aug = torch.cat((current_image_depth, current_image_depth_flip), axis=0)                  

            B, _, H, W = current_image_depth.size()             
            
            with torch.no_grad():
                proj_3d, outputs = model_depth.forward(current_image_depth_aug) #for depth360   

            proj_3d_resize = transform_3d(proj_3d)                
            #batch_3d_point = proj_3d.to(device)
            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device) 
            
            image_list, ped_list = model_pedest.forward(255.0*current_image_raw, 255.0*current_image_depth, proj_3d)
            """    
            # Logging            
            loss = torch.tensor(0.0)
            dist_loss = torch.tensor(0.0)
            diff_loss = torch.tensor(0.0)
            loss_geo = torch.tensor(0.0)
                                    
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"smooth_loss": diff_loss.item()})
            wandb.log({"geo_loss": loss_geo.item()})        
            
            if i % 1000 == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss.item()
                losses['smooth_loss'] = diff_loss.item()                 
                losses['geo_loss'] = loss_geo.item()                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
                """
                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
                """
            B, = id_num.size()     
            for i in range(B):
                idn = id_num[i].item()
                yaw_ang = np.arctan2(relative_mat[i, 1, 0].item(), relative_mat[i, 1, 1].item())
                yaw.append([idn, yaw_ang])
                
                #save_dict = {}
                #save_dict["id_num"] = id_num[i].item()
                """
                save_dict["filtered_position"] = filtered_position[i].numpy()
                save_dict["filtered_heading"] = filtered_heading[i].numpy()                        
                save_dict["utm_position"] = utm_position[i].numpy()
                save_dict["compass_heading"] = compass_heading[i].numpy()    
                save_dict["gyroscope"] = gyroscope[i].numpy()
                save_dict["accelerometer"] = accelerometer[i].numpy() 
                save_dict["wheel_rpm"] = wheel_rpm[i].numpy()
                save_dict["action"] = action[i].numpy() 
                save_dict["action_original"] = action_original[i].numpy()
                save_dict["episode_index"] = episode_index[i].item() 
                save_dict["frame_index"] = frame_index[i].item()
                save_dict["timestamp"] = timestamp[i].item()    
                                                                                   
                save_dict["utm_zone_letter"] = utm_zone_letter[i].item()
                save_dict["utm_zone_number"] = utm_zone_number[i].item() 
                save_dict["video_id_number_1"] = video_id_number_1[i].item()
                save_dict["video_id_number_2"] = video_id_number_2[i].item()  
                """
                """
                save_dict["relative_mat"] = relative_mat[i].numpy()                             
                with open("/mnt/ephemeral2/noriaki/frodobots_dataset/obs2/" + str(int(idn)) + ".pkl", 'wb') as file:
                    pickle.dump(save_dict, file)
                """
            """
            for i in range(B):
                idn = id_num[i].item()
                proj_3d_save = proj_3d_resize[i].cpu().to(torch.float32).numpy()
                with open("/mnt/ephemeral2/noriaki/frodobots_dataset/ped_est/" + str(int(idn)) + ".pkl", 'wb') as file:
                    pickle.dump(ped_list[i], file)
            """        
            """
                with open("/media/noriaki/Noriaki_Data3/frodobots_dataset/3d_point/" + str(int(idn)) + ".pkl", 'wb') as file:
                    pickle.dump(proj_3d_save.astype(np.float16), file)
            """
            """                
            if image_log_freq != 0:# and i % image_log_freq == 0:
                visualize_annotate_estimation(
                    image_list, #current_image_depth, #image_list, #current_image_depth,
                    image_list,
                    batch_3d_point,
                    ped_list,
                    #goal_pos,
                    #local_yaw,
                    #linear_vel.cpu(),
                    #angular_vel.cpu(),
                    #last_poses.cpu(),
                    #rsize.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
            """  
        with open("/mnt/ephemeral2/noriaki/frodobots_dataset/yaw_10k.pkl", 'wb') as file:
            pickle.dump(yaw, file)
                        
def evaluate_annotate(
    eval_type: str,
    #ema_model: EMAModel,
    #optimizer: Adam,
    dataloader: DataLoader,
    #transform: transforms,
    device: torch.device,
    #noise_scheduler: DDPMScheduler,
    #goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    model_depth,
    model_pedest,
    #device2,         
    #len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    #ema_model = ema_model.averaged_model    
    #ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth_loss", "test", window_size=print_log_freq)
    geo_loss_logger = Logger("geo_loss", "test", window_size=print_log_freq)

    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "smooth_loss": smooth_loss_logger,
        "geo_loss": geo_loss_logger,
    }

    #D = np.array(np.load('/mnt/ephemeral2/Learning-to-Drive-Anywhere-via-MBRA/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    #K = np.array(np.load('/mnt/ephemeral2/Learning-to-Drive-Anywhere-via-MBRA/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))

    mask_360 = np.loadtxt(open("/mnt/ephemeral2/Learning-to-Drive-Anywhere-via-MBRA/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device)
    transform_3d = T.Resize(size = (96,96))  
        
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,                   
                _,
                _,
                #_, 
                id_num,
                image_raw,
                _,
                _,
            ) = data
            #print("test", id_num)
            #at different GPUs
            #print(current_image.size())
            current_image_depth = (current_image.to(device))*mask_360_torch
            current_image_raw = image_raw.to(device)
            
            current_image_depth_flip = torch.flip(current_image_depth, dims=[3]) 
            current_image_depth_aug = torch.cat((current_image_depth, current_image_depth_flip), axis=0)                  

            B, _, H, W = current_image_depth.size()             
            
            with torch.no_grad():
                proj_3d, outputs = model_depth.forward(current_image_depth_aug) #for depth360   

            proj_3d_resize = transform_3d(proj_3d)                
            #batch_3d_point = proj_3d.to(device)
            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device) 
            
            image_list, ped_list = model_pedest.forward(255.0*current_image_raw, 255.0*current_image_depth, proj_3d)
                        
            """                        
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            rsize = torch.rand(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            #rsize = 0.5*torch.ones(B, 1, 1).to(device)          
            
            with torch.no_grad():
                linear_vel, angular_vel = ema_model(obs_image, goal_image, rsize)
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((pz_ref.unsqueeze(1), -px_ref.unsqueeze(1)), axis=1) #from camera coordinate to robot local coordinate
       
            mat_1 = torch.cat((torch.cos(ry_ref).unsqueeze(1), -torch.sin(ry_ref).unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
            mat_2 = torch.cat((torch.sin(ry_ref).unsqueeze(1), torch.cos(ry_ref).unsqueeze(1), -px_ref.unsqueeze(1)), axis=1)
            mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
            last_pose_mat = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)            
            
            #print("local_goal_mat", local_goal_mat)
            #print("last_pose_mat", last_pose_mat)
                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 

            PC3D = []
            for j in range(len_traj_pred):
                px_ref = px_ref_list[j]
                pz_ref = pz_ref_list[j]
                ry_ref = ry_ref_list[j]                

                Tod = torch.zeros((B, 4, 4)).to(device)
                Tod[:, 0, 0] = torch.cos(ry_ref)
                Tod[:, 0, 2] = torch.sin(ry_ref)
                Tod[:, 1, 1] = 1.0
                Tod[:, 2, 0] = -torch.sin(ry_ref)
                Tod[:, 2, 2] = torch.cos(ry_ref)
                Tod[:, 0, 3] = px_ref
                Tod[:, 2, 3] = pz_ref
                Tod[:, 3, 3] = 1.0

                Ttrans = torch.inverse(Tod)[:, :3, :]               
                batch_3d_point_x = torch.cat((batch_3d_point.view(B, 3, -1), torch.ones(B,1,416*128).to(device)), axis=1)
                cam_points_trans = torch.matmul(Ttrans, batch_3d_point_x).view(B, 3, 128, 416)
                PC3D.append(cam_points_trans.unsqueeze(1))                                                  
            
            PC3D_cat = torch.cat(PC3D, axis=1)                    
  
            loss_geo = geometry_criterion(PC3D_cat, rsize[:,:,0], len_traj_pred, device)
            #loss_geo = torch.tensor(0.0)
            
            loss = 1.0*dist_loss + 1.0*diff_loss + 2.0*loss_geo
            """
            # Logging            
            loss = torch.tensor(0.0)
            dist_loss = torch.tensor(0.0)
            diff_loss = torch.tensor(0.0)
            loss_geo = torch.tensor(0.0)                                    
            loss_cpu = loss.cpu().item()
            #print(dist_loss, diff_loss, loss_cpu)            
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"smooth_loss": diff_loss.item()})
            wandb.log({"geo_loss": loss_geo.item()})

            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss.item()
                losses['smooth_loss'] = diff_loss.item()                 
                losses['geo_loss'] = loss_geo.item()
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            """
            for i in range(B):
                idn = id_num[i].item()
                proj_3d_save = proj_3d_resize[i].cpu().to(torch.float16).numpy()
                with open("/media/noriaki/Noriaki_Data3/frodobots_dataset/ped_est/" + str(int(idn)) + ".pkl", 'wb') as file:
                    pickle.dump(ped_list[i], file)
                with open("/media/noriaki/Noriaki_Data3/frodobots_dataset/3d_point/" + str(int(idn)) + ".pkl", 'wb') as file:
                    pickle.dump(proj_3d_save.astype(np.float16), file)
            """
            """
            if image_log_freq != 0:# and i % image_log_freq == 0:
                visualize_annotate_estimation(
                    image_list, #current_image_depth, #image_list, #current_image_depth,
                    image_list,
                    batch_3d_point,
                    ped_list,
                    #goal_pos,
                    #local_yaw,
                    #linear_vel.cpu(),
                    #angular_vel.cpu(),
                    #last_poses.cpu(),
                    #rsize.cpu(),
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
            """

def evaluate_lelan(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    """
    ema_model.eval()    
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", eval_type, window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", eval_type, window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", eval_type, window_size=print_log_freq)     
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }    
    num_batches = max(int(num_batches * eval_fraction), 1)

    all_total = 0.0
    all_dist = 0.0
    all_diff = 0.0
    
    count_batch = 0
    data_size = 0
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                obj_poses,
                obj_inst,
                goal_pos_norm,
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]       

            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])                         
            batch_obs_images = transform(obs_image).to(device)
            batch_obj_poses = obj_poses.to(device)
            
            B = batch_obs_images.shape[0]
            with torch.no_grad():
                batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
                feat_text = ema_model("text_encoder", inst_ref=batch_obj_inst)                  
                obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32))
                linear_vel, angular_vel = ema_model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                
                px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
                px_ref = px_ref_list[-1]
                pz_ref = pz_ref_list[-1]
                ry_ref = ry_ref_list[-1]
                                                    
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
                        
            dist_loss = nn.functional.mse_loss(last_poses, batch_obj_poses)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
                        
            # Logging
            loss_cpu = dist_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"total_eval_loss": (dist_loss + 1.0*diff_loss).item()})
            wandb.log({"dist_eval_loss": dist_loss.item()})
            wandb.log({"diff_eval_loss": diff_loss.item()})

            all_total += (dist_loss + 1.0*diff_loss).item()
            all_dist += dist_loss.item()
            all_diff += diff_loss.item()
            count_batch += 1.0
            data_size += B
            if i % print_log_freq == 0 and print_log_freq != 0: 
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()             
                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    obj_poses,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )                
    print(eval_type, "total loss:", all_total/count_batch, "dist loss:", all_dist/count_batch, "diff loss:", all_diff/count_batch, "batch count:", count_batch, "data size:", data_size)

def evaluate_lelan_col(
    eval_type: str,
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,    
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    weight_col_loss: float,    
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        ema_model_nomad (nn.Module): exponential moving average version of pre-trained NoMaD policy
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        weight_col_loss (float) : weight for collision avoidance loss 
        epoch (int): current epoch    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    """

    ema_model.eval()
    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()       
    num_batches = len(dataloader)

    total_loss_logger = Logger("total loss", eval_type, window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", eval_type, window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", eval_type, window_size=print_log_freq)    
    col_loss_logger = Logger("col loss", eval_type, window_size=print_log_freq) 
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
        "col loss": col_loss_logger,        
    }    
    num_batches = max(int(num_batches * eval_fraction), 1)

    all_total = 0.0
    all_dist = 0.0
    all_diff = 0.0
    all_col = 0.0
        
    count_batch = 0
    data_size = 0
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                goal_pos,
                obj_inst,
                goal_pos_norm,
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((255.0*obs_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((255.0*goal_image).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])                                                       
            batch_obs_current = transform(obs_image).to(device)
            batch_goal_pos = goal_pos.to(device)
            goal_pos_norm = goal_pos_norm.to(device)                              
            batch_obs_images = [transform(TF.resize(obs, (96, 96), antialias=True)) for obs in obs_images_list]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(TF.resize(goal_image, (96, 96), antialias=True)).to(device)
            
            B = batch_obs_images.shape[0]
                        
            # split into batches
            batch_obs_images_list = torch.split(batch_obs_images, B, dim=0)
            batch_goal_images_list = torch.split(batch_goal_images, B, dim=0)

            with torch.no_grad():
                select_traj = supervision_from_nomad(
                    ema_model_nomad,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos_norm,
                    device,
                    project_folder,
                    epoch,
                    B,
                    i,                
                    30,
                    use_wandb,
                    )                
            
            with torch.no_grad():
                batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)         
                feat_text = ema_model("text_encoder", inst_ref=batch_obj_inst)       
                                
                B = batch_obs_images.shape[0]

                obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32), current_img=batch_obs_current)
                linear_vel, angular_vel = ema_model("dist_pred_net", obsgoal_cond=obsgoal_cond)

                px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
                px_ref = px_ref_list[-1]
                pz_ref = pz_ref_list[-1]
                ry_ref = ry_ref_list[-1]
                                                    
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
            px_ref_listx = []
            pz_ref_listx = []
            for it in range(8):            
                px_ref_listx.append(px_ref_list[it].unsqueeze(1).unsqueeze(2))
                pz_ref_listx.append(pz_ref_list[it].unsqueeze(1).unsqueeze(2))
            traj_policy = torch.concat((torch.concat(pz_ref_listx, axis=1), -torch.concat(px_ref_listx, axis=1)), axis=2)
                                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 

            mask_nomad = (batch_goal_pos[:,1:2] > 1.0).float().unsqueeze(1).repeat(1,8,2)
            mask_dist = (~(batch_goal_pos[:,1:2] > 1.0)).float()
            sum_dist = mask_dist.sum()            
            col_loss = nn.functional.mse_loss(mask_nomad*traj_policy, 0.12*mask_nomad*select_traj)*float(B)/(float(B) - sum_dist.float() + 1e-7) #0.12 is de-normalization
            
            loss = 1.0*dist_loss + 1.0*diff_loss + weight_col_loss*col_loss
                                                
            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"total_eval_loss": (dist_loss + 1.0*diff_loss + weight_col_loss*col_loss).item()})
            wandb.log({"dist_eval_loss": dist_loss.item()})
            wandb.log({"diff_eval_loss": diff_loss.item()})
            wandb.log({"col_eval_loss": col_loss.item()})
            
            all_total += (dist_loss + 1.0*diff_loss + weight_col_loss*col_loss).item()
            all_dist += dist_loss.item()
            all_diff += diff_loss.item()
            all_col += col_loss.item()            
            count_batch += 1.0
            data_size += B
            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()             
                losses['col loss'] = col_loss.item()       
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_col_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    (0.12*select_traj).cpu(),                    
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )                
    print(eval_type, "total loss:", all_total/count_batch, "dist loss:", all_dist/count_batch, "diff loss:", all_diff/count_batch, "col loss:", all_col/count_batch, "batch count:", count_batch, "data size:", data_size)

def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            ### RANDOM MASK ERROR ###
            # Predict the noise residual
            rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
            
            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
            
            ### NO MASK ERROR ###
            # Predict the noise residual
            no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
            
            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
            
            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
            
            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
            wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
            wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )

###
def evaluate_exaug_dist_gnm_delay(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    model_depth,
    #model_pedtraj,
    device2,         
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    distall_loss_logger = Logger("distall_loss", "test", window_size=print_log_freq)    
    smooth_loss_logger = Logger("smooth_loss", "test", window_size=print_log_freq)
    geo_loss_logger = Logger("geo_loss", "test", window_size=print_log_freq)
    social_loss_logger = Logger("social_loss", "test", window_size=print_log_freq)
    personal_loss_logger = Logger("personal_loss", "test", window_size=print_log_freq)
    disttemp_loss_logger = Logger("disttemp_loss", "test", window_size=print_log_freq)
    goal_dist_mean_logger = Logger("goal_dist_mean", "test", window_size=print_log_freq)
    goal_dist_median_logger = Logger("goal_dist_median", "test", window_size=print_log_freq)
        
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "distall_loss": distall_loss_logger,        
        "smooth_loss": smooth_loss_logger,
        "geo_loss": geo_loss_logger,
        "social_loss": social_loss_logger,
        "personal_loss": personal_loss_logger,     
        "disttemp_loss": disttemp_loss_logger,   
        "goal_dist_mean": goal_dist_mean_logger,   
        "goal_dist_median": goal_dist_median_logger,                    
    }
    
    mask_360 = np.loadtxt(open("./mask_360view.csv", "rb"), delimiter=",", skiprows=0)   
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)

    linear_vel_old = 0.5*torch.rand(100, 8).float().to(device)
    angular_vel_old = 1.0*torch.rand(100, 8).float().to(device)
    
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
                current_image_depth,
                geoloss_range,
                local_goal_mat,
                local_yaw,
            ) = data         
            B, _, H, W = current_image_depth.size()    
            current_image_depth = current_image_depth.to(device2)                          
            obs_images = torch.split(obs_image, 3, dim=1)
                        
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)           
            
            with torch.no_grad():
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   

            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device) 

            # Get distance label
            distance_metric = torch.sqrt(goal_pos.to(device)[:,0]**2 + goal_pos.to(device)[:,1]**2)
            fargoal_mask = ((torch.abs(local_goal_mat[:, 0,2]) < 2.0) * (torch.abs(local_goal_mat[:, 1,2]) < 2.0)).to(device)
            
            distance = dist_label.float().to(device)
            goal_mask = (distance > 0.1) * fargoal_mask
            goal_mask_zero = distance > 0.1

            for ig in range(B):
                if not goal_mask_zero[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            rsize = torch.rand(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
            delay = torch.randint(0, 5, (B, 1, 1)).to(device)      

            cs = random.randint(0,2)            
            vel_past = torch.cat((linear_vel_old[:, cs:cs+6], angular_vel_old[:, cs:cs+6]), axis=1).unsqueeze(2)     
            
            with torch.no_grad():
                linear_vel, angular_vel, dist_temp = ema_model(obs_image, goal_image, rsize, delay, vel_past)

            for ig in range(B):
                linear_vel_old[ig, delay[ig,0,0]:6] *= 0.0
                angular_vel_old[ig, delay[ig,0,0]:6] *= 0.0
                                
            linear_vel_d = torch.cat((linear_vel_old, linear_vel), axis=1)
            angular_vel_d = torch.cat((angular_vel_old, angular_vel), axis=1)            
            
            linear_vel_old = linear_vel.detach()
            angular_vel_old = angular_vel.detach()                
                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel_d, angular_vel_d)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((pz_ref.unsqueeze(1), -px_ref.unsqueeze(1)), axis=1) #from camera coordinate to robot local coordinate
       
            mat_1 = torch.cat((torch.cos(-ry_ref).unsqueeze(1), -torch.sin(-ry_ref).unsqueeze(1), 2.0*pz_ref.unsqueeze(1)), axis=1)
            mat_2 = torch.cat((torch.sin(-ry_ref).unsqueeze(1), torch.cos(-ry_ref).unsqueeze(1), -2.0*px_ref.unsqueeze(1)), axis=1)
            mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
            last_pose_mat = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)            
            
            robot_traj_list = []
            for ip in range(len(px_ref_list)):
                mat_1 = torch.cat((torch.cos(-ry_ref_list[ip]).unsqueeze(1), -torch.sin(-ry_ref_list[ip]).unsqueeze(1), 2.0 * pz_ref_list[ip].unsqueeze(1)), axis=1)
                mat_2 = torch.cat((torch.sin(-ry_ref_list[ip]).unsqueeze(1), torch.cos(-ry_ref_list[ip]).unsqueeze(1), -2.0 * px_ref_list[ip].unsqueeze(1)), axis=1)
                mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
                mat_combine = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
                robot_traj_list.append(mat_combine.unsqueeze(1))
            robot_traj_vec = torch.cat(robot_traj_list, axis=1)

            local_goal_mat[:, 0,2] *= 2.0                    
            local_goal_mat[:, 1,2] *= 2.0  
            local_goal_vec = local_goal_mat.unsqueeze(1).repeat(1,8,1,1)

            dist_loss = nn.functional.mse_loss(last_pose_mat[goal_mask], local_goal_mat.to(device)[goal_mask])                                    
            distall_loss = nn.functional.mse_loss(robot_traj_vec[goal_mask, 6:14], local_goal_vec.to(device)[goal_mask])                     
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1][goal_mask], linear_vel[:,1:][goal_mask]) + nn.functional.mse_loss(angular_vel[:,:-1][goal_mask], angular_vel[:,1:][goal_mask]) 

            # Get distance label
            distance = distance.float().to(device)

            # Predict distance         
            dist_temp_loss = F.mse_loss(dist_temp.squeeze(-1), distance)

            PC3D = []
            for j in range(len_traj_pred):
                px_ref = px_ref_list[j+6]
                pz_ref = pz_ref_list[j+6]
                ry_ref = ry_ref_list[j+6]                

                Tod = torch.zeros((B, 4, 4)).to(device)
                Tod[:, 0, 0] = torch.cos(ry_ref)
                Tod[:, 0, 2] = torch.sin(ry_ref)
                Tod[:, 1, 1] = 1.0
                Tod[:, 2, 0] = -torch.sin(ry_ref)
                Tod[:, 2, 2] = torch.cos(ry_ref)
                Tod[:, 0, 3] = px_ref
                Tod[:, 2, 3] = pz_ref
                Tod[:, 3, 3] = 1.0

                Ttrans = torch.inverse(Tod)[:, :3, :]               
                batch_3d_point_x = torch.cat((batch_3d_point.view(B, 3, -1), torch.ones(B,1,416*128).to(device)), axis=1)
                cam_points_trans = torch.matmul(Ttrans, batch_3d_point_x).view(B, 3, 128, 416)
                PC3D.append(cam_points_trans.unsqueeze(1))                                                  
            
            PC3D_cat = torch.cat(PC3D, axis=1)                    
  
            loss_geo = geometry_criterion_range(PC3D_cat[goal_mask], rsize[:,:,0][goal_mask], len_traj_pred, geoloss_range.to(device)[goal_mask], device)
            norm = 10.0
            """
            if sacson:
                ped_past = torch.cat((-torch.flip(ped_list[:,:,1], dims=[1]), torch.flip(ped_list[:,:,0], dims=[1])), axis=1).to(device)  
                robot_past = torch.cat((-torch.flip(robot_list[:,:,1], dims=[1]), torch.flip(robot_list[:,:,0], dims=[1])), axis=1).to(device)
                
                robot_future_x = []
                robot_future_z = []
                for ir in range(len(px_ref_list)):
                    robot_future_x.append(px_ref_list[ir].unsqueeze(1))
                    robot_future_z.append(pz_ref_list[ir].unsqueeze(1))                    
                robot_future = torch.cat((torch.cat(robot_future_x, axis=1), torch.cat(robot_future_z, axis=1)), axis=1).to(device)  
                            
                ped_past_c = torch.clamp(ped_past, min=-10.0, max=10.0)/norm
                robot_past_c = torch.clamp(robot_past, min=-10.0, max=10.0)/norm
                robot_future_c = torch.clamp(robot_future, min=-10.0, max=10.0)/norm
                
                flag_ped = (ped_past_c[:,0] != 0.0) & (ped_past_c[:,8] != 0.0)
                
                robot_past_c = 0.333*robot_past_c #frodbot is much faster than Vizbot.
                delta_est_ped_traj = model_pedtraj(ped_past_c, robot_past_c, robot_future_c)
                delta_est_ped_traj_zero = model_pedtraj(ped_past_c, robot_past_c, 0.0*robot_future_c)

                traj_x = torch.cumsum(delta_est_ped_traj[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_y = torch.cumsum(delta_est_ped_traj[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj = torch.clamp(torch.cat((traj_x, traj_y), axis=1), min=-10.0, max=10.0)

                delta_est_ped_traj_zero_detach = delta_est_ped_traj_zero.detach()
                traj_xz = torch.cumsum(delta_est_ped_traj_zero_detach[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_yz = torch.cumsum(delta_est_ped_traj_zero_detach[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj_zeros = torch.clamp(torch.cat((traj_xz, traj_yz), axis=1), min=-10.0, max=10.0)
                
                social_loss = nn.functional.mse_loss(est_ped_traj[flag_ped*goal_mask], est_ped_traj_zeros[flag_ped*goal_mask])
                
                max_pl = rsize.squeeze(1).repeat(1,8) + 0.5
                min_pl = rsize.squeeze(1).repeat(1,8) * 0.0
                personal_loss = nn.functional.mse_loss(max_pl[flag_ped*goal_mask], torch.clamp(torch.sqrt((norm*robot_future_c[flag_ped*goal_mask][:,0:8] - norm*est_ped_traj[flag_ped*goal_mask][:,0:8])**2 + (norm*robot_future_c[flag_ped*goal_mask][:,8:16] - norm*est_ped_traj[flag_ped*goal_mask][:,8:16])**2), min=min_pl[flag_ped*goal_mask], max= max_pl[flag_ped*goal_mask]))
            else:
                est_ped_traj = torch.zeros(B, 16)
                est_ped_traj_zeros = torch.zeros(B, 16)      
                ped_past_c = torch.zeros(B, 16)
                robot_past_c = torch.zeros(B, 16)
                social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
            """
            est_ped_traj = torch.zeros(B, 16)
            est_ped_traj_zeros = torch.zeros(B, 16)      
            ped_past_c = torch.zeros(B, 16)
            robot_past_c = torch.zeros(B, 16)
            social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
            personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                            
            loss = 4.0*dist_loss + 0.4*distall_loss + 0.5*diff_loss + 10.0*loss_geo + 100.0*social_loss + 10.0*personal_loss + 0.001*dist_temp_loss 
            goal_dist_mean = torch.max(torch.sqrt(goal_pos.to(device)[:,0][goal_mask]**2 + goal_pos.to(device)[:,1][goal_mask]**2))
            goal_dist_median = torch.min(torch.sqrt(goal_pos.to(device)[:,0][goal_mask]**2 + goal_pos.to(device)[:,1][goal_mask]**2))
            # Logging            
            loss_cpu = loss.cpu().item()          
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"distall_loss": distall_loss.item()})            
            wandb.log({"smooth_loss": diff_loss.item()})
            wandb.log({"geo_loss": loss_geo.item()})
            wandb.log({"social_loss": social_loss.item()})
            wandb.log({"personal_loss": personal_loss.item()})
            wandb.log({"disttemp_loss": dist_temp_loss.item()})
            wandb.log({"goal_dist_mean": goal_dist_mean.item()})
            wandb.log({"goal_dist_median": goal_dist_median.item()})
                                    
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss.item()
                losses['distall_loss'] = distall_loss.item()                
                losses['smooth_loss'] = diff_loss.item()                 
                losses['geo_loss'] = loss_geo.item()
                losses['social_loss'] = social_loss.item()                 
                losses['personal_loss'] = personal_loss.item()  
                losses['disttemp_loss'] = dist_temp_loss.item()  
                losses['goal_dist_mean'] = goal_dist_mean.item()  
                losses['goal_dist_median'] = goal_dist_median.item()  
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_exaug_delay_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    batch_3d_point,
                    goal_pos,
                    local_yaw,
                    linear_vel_d.cpu(),
                    angular_vel_d.cpu(),
                    norm*ped_past_c.cpu(),
                    norm*est_ped_traj.cpu(),
                    norm*est_ped_traj_zeros.cpu(),
                    norm*robot_past_c.cpu(),
                    last_poses.cpu(),
                    rsize.cpu(),
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )    
###
def evaluate_il_dist_gnm(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,
    #model_depth,
    #model_pedtraj,
    #device2,         
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,                     
    }
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break                    
            B, _, H, W = obs_image.size()  
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            
            action_mask = action_mask.to(device)
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)
            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)      
            
            with torch.no_grad():
                dist_pred, action_pred = ema_model(obs_image, goal_image)
            
            losses = _compute_losses(
                dist_label=distance,
                action_label=actions,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=0.5,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )                  
                  
###
def evaluate_il_dist_gnm_gps(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool,   
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                     
    }

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    goal_yaw,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break      
            B, _, H, W = obs_image.size()  
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            actions = actions.to(device)
            action_mask = action_mask.to(device)
            
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)

            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask
            goal_pose_gps = torch.cat((goal_pos, torch.cos(goal_yaw).unsqueeze(1), torch.sin(goal_yaw).unsqueeze(1)), axis=1).to(device)

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)   
            
            with torch.no_grad():
                action_pred = ema_model(obs_image, goal_pose_gps)

            losses = _compute_losses_gps(
                action_label=actions,
                action_pred=action_pred,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()          
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation_gps(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    actions,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )   
###
def evaluate_il_dist_gnm_gps_map(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    epoch: int,
    sacson: bool, 
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "action_loss": action_loss_logger,                     
    }
    
    mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)    
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch): 
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    goal_yaw,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break                   
            B, _, H, W = obs_image.size()  
            current_map_image = torch.zeros(B, 3, 96, 96)
            goal_map_image = torch.zeros(B, 3, 96, 96)            
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image_map = obs_images[0].to(device)        
            obs_image = torch.cat(obs_images, dim=1)
            
            map_images = torch.cat((current_map_image.to(device), goal_map_image.to(device), obs_image_map), axis=1)                
            actions = actions.to(device)           
            action_mask = action_mask.to(device)           
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)

            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask
            goal_pose_gps = torch.cat((goal_pos, torch.cos(goal_yaw).unsqueeze(1), torch.sin(goal_yaw).unsqueeze(1)), axis=1).to(device)

            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            #model calculate            
            goal_mask_sub = torch.zeros(B).to(device) #GNM: Goal pose only            
            with torch.no_grad():
                action_pred, _ = ema_model(obs_image, goal_pose_gps, map_images, goal_mask_sub)
            
            losses = _compute_losses_gps(
                action_label=actions,
                action_pred=action_pred,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            action_loss_cpu = losses["action_loss"].item()           
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation_gps(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    actions,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )

def evaluate_il_dist_gnm_gps_map2(
    eval_type: str,
    ema_model: EMAModel,
    #optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    #noise_scheduler: DDPMScheduler,
    #goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    sacson: bool,
    #model_depth,
    #model_pedtraj,
    #device2,         
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,                     
    }
    """
    D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))

    mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)    
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    """
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            """
            (
                obs_image, 
                goal_image,
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,                   
                _,
                _,
                #_, 
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,                
            ) = data        
            """  
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    goal_yaw,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break   
            """
            current_image_depth = (current_image.to(device2))*mask_360_torch
            B, _, H, W = current_image_depth.size()             
            
            with torch.no_grad():
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   
                
            #batch_3d_point = proj_3d.to(device)
            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device) 
            """                      
            B, _, H, W = obs_image.size()  

            current_map_image = torch.zeros(B, 3, 96, 96)
            goal_map_image = torch.zeros(B, 3, 96, 96)
            
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image_map = obs_images[0].to(device)        
            obs_image = torch.cat(obs_images, dim=1)
            
            map_images = torch.cat((current_map_image.to(device), goal_map_image.to(device), obs_image_map), axis=1)                
            actions = actions.to(device)
            #print("actions", actions.size())
            
            action_mask = action_mask.to(device)
            
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)

            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask
            goal_pose_gps = torch.cat((goal_pos, torch.cos(goal_yaw).unsqueeze(1), torch.sin(goal_yaw).unsqueeze(1)), axis=1).to(device)
            # Get distance label
            #distance = distance.float().to(device)
            #goal_mask = distance > 0.1
                        
            #print(goal_mask)
            """
            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]
            """
            goal_mask_gnm = []
            for idg in range(B):
                if distance[idg] <= 20:
                    goal_mask_gnm.append(random.randint(4,6))
                else:
                    goal_mask_gnm.append(random.randint(4,5))
            goal_mask_sub = torch.tensor(goal_mask_gnm).to(device)                 
            
            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            #model calculate 
            #goal_mask_f = torch.randint(0,3,(B,)) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            #goal_mask_sub = torch.zeros(B).to(device) #GNM: Goal pose only
            #goal_mask = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            
            with torch.no_grad():
                action_pred, dist_pred, _ = ema_model(obs_image, goal_pose_gps, map_images, goal_image, goal_mask_sub)
                #linear_vel, angular_vel, dist_temp = ema_model(obs_image, goal_image, rsize)

            #print("distance", distance)
            #print("dist_pred", dist_pred)
            #dist_loss = F.mse_loss(dist_pred.squeeze(-1), distance.float())
            #print(dist_loss)
            """
            losses = _compute_losses_gps(
                action_label=actions,
                action_pred=action_pred,
                learn_angle=True,
                action_mask=action_mask,
            )
            """
            
            losses = _compute_losses(
                dist_label=distance,
                action_label=actions,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=0.5,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            """                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((pz_ref.unsqueeze(1), -px_ref.unsqueeze(1)), axis=1) #from camera coordinate to robot local coordinate
       
            mat_1 = torch.cat((torch.cos(-ry_ref).unsqueeze(1), -torch.sin(-ry_ref).unsqueeze(1), 2.0*pz_ref.unsqueeze(1)), axis=1)
            mat_2 = torch.cat((torch.sin(-ry_ref).unsqueeze(1), torch.cos(-ry_ref).unsqueeze(1), -2.0*px_ref.unsqueeze(1)), axis=1)
            mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
            last_pose_mat = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)            
            
            robot_traj_list = []
            for ip in range(len(px_ref_list)):
                mat_1 = torch.cat((torch.cos(-ry_ref_list[ip]).unsqueeze(1), -torch.sin(-ry_ref_list[ip]).unsqueeze(1), 2.0 * pz_ref_list[ip].unsqueeze(1)), axis=1)
                mat_2 = torch.cat((torch.sin(-ry_ref_list[ip]).unsqueeze(1), torch.cos(-ry_ref_list[ip]).unsqueeze(1), -2.0 * px_ref_list[ip].unsqueeze(1)), axis=1)
                mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
                mat_combine = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
                robot_traj_list.append(mat_combine.unsqueeze(1))
            robot_traj_vec = torch.cat(robot_traj_list, axis=1)

            local_goal_mat[:, 0,2] *= 2.0                    
            local_goal_mat[:, 1,2] *= 2.0  
            local_goal_vec = local_goal_mat.unsqueeze(1).repeat(1,8,1,1)

            dist_loss = nn.functional.mse_loss(last_pose_mat[goal_mask], local_goal_mat.to(device)[goal_mask])                                    
            #dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos) 
            distall_loss = nn.functional.mse_loss(robot_traj_vec[goal_mask], local_goal_vec.to(device)[goal_mask])                     
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1][goal_mask], linear_vel[:,1:][goal_mask]) + nn.functional.mse_loss(angular_vel[:,:-1][goal_mask], angular_vel[:,1:][goal_mask]) 

            # Get distance label
            distance = distance.float().to(device)

            # Predict distance         
            dist_temp_loss = F.mse_loss(dist_temp.squeeze(-1), distance)

            PC3D = []
            for j in range(len_traj_pred):
                px_ref = px_ref_list[j]
                pz_ref = pz_ref_list[j]
                ry_ref = ry_ref_list[j]                

                Tod = torch.zeros((B, 4, 4)).to(device)
                Tod[:, 0, 0] = torch.cos(ry_ref)
                Tod[:, 0, 2] = torch.sin(ry_ref)
                Tod[:, 1, 1] = 1.0
                Tod[:, 2, 0] = -torch.sin(ry_ref)
                Tod[:, 2, 2] = torch.cos(ry_ref)
                Tod[:, 0, 3] = px_ref
                Tod[:, 2, 3] = pz_ref
                Tod[:, 3, 3] = 1.0

                Ttrans = torch.inverse(Tod)[:, :3, :]               
                batch_3d_point_x = torch.cat((batch_3d_point.view(B, 3, -1), torch.ones(B,1,416*128).to(device)), axis=1)
                cam_points_trans = torch.matmul(Ttrans, batch_3d_point_x).view(B, 3, 128, 416)
                PC3D.append(cam_points_trans.unsqueeze(1))                                                  
            
            PC3D_cat = torch.cat(PC3D, axis=1)                    
  
            loss_geo = geometry_criterion(PC3D_cat[goal_mask], rsize[:,:,0][goal_mask], len_traj_pred, device)
            norm = 10.0
            if sacson:
                ped_past = torch.cat((-torch.flip(ped_list[:,:,1], dims=[1]), torch.flip(ped_list[:,:,0], dims=[1])), axis=1).to(device)  
                robot_past = torch.cat((-torch.flip(robot_list[:,:,1], dims=[1]), torch.flip(robot_list[:,:,0], dims=[1])), axis=1).to(device)
                
                robot_future_x = []
                robot_future_z = []
                for ir in range(len(px_ref_list)):
                    robot_future_x.append(px_ref_list[ir].unsqueeze(1))
                    robot_future_z.append(pz_ref_list[ir].unsqueeze(1))                    
                robot_future = torch.cat((torch.cat(robot_future_x, axis=1), torch.cat(robot_future_z, axis=1)), axis=1).to(device)  
                            
                ped_past_c = torch.clamp(ped_past, min=-10.0, max=10.0)/norm
                robot_past_c = torch.clamp(robot_past, min=-10.0, max=10.0)/norm
                robot_future_c = torch.clamp(robot_future, min=-10.0, max=10.0)/norm
                
                flag_ped = (ped_past_c[:,0] != 0.0) & (ped_past_c[:,8] != 0.0)
                
                robot_past_c = 0.333*robot_past_c #frodbot is much faster than Vizbot.
                delta_est_ped_traj = model_pedtraj(ped_past_c, robot_past_c, robot_future_c)
                delta_est_ped_traj_zero = model_pedtraj(ped_past_c, robot_past_c, 0.0*robot_future_c)

                traj_x = torch.cumsum(delta_est_ped_traj[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_y = torch.cumsum(delta_est_ped_traj[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj = torch.clamp(torch.cat((traj_x, traj_y), axis=1), min=-10.0, max=10.0)

                delta_est_ped_traj_zero_detach = delta_est_ped_traj_zero.detach()
                traj_xz = torch.cumsum(delta_est_ped_traj_zero_detach[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_yz = torch.cumsum(delta_est_ped_traj_zero_detach[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj_zeros = torch.clamp(torch.cat((traj_xz, traj_yz), axis=1), min=-10.0, max=10.0)
                
                social_loss = nn.functional.mse_loss(est_ped_traj[flag_ped*goal_mask], est_ped_traj_zeros[flag_ped*goal_mask])
                
                max_pl = rsize.squeeze(1).repeat(1,8) + 0.5
                min_pl = rsize.squeeze(1).repeat(1,8) * 0.0
                personal_loss = nn.functional.mse_loss(max_pl[flag_ped*goal_mask], torch.clamp(torch.sqrt((norm*robot_future_c[flag_ped*goal_mask][:,0:8] - norm*est_ped_traj[flag_ped*goal_mask][:,0:8])**2 + (norm*robot_future_c[flag_ped*goal_mask][:,8:16] - norm*est_ped_traj[flag_ped*goal_mask][:,8:16])**2), min=min_pl[flag_ped*goal_mask], max= max_pl[flag_ped*goal_mask]))
            else:
                est_ped_traj = torch.zeros(B, 16)
                est_ped_traj_zeros = torch.zeros(B, 16)      
                ped_past_c = torch.zeros(B, 16)
                robot_past_c = torch.zeros(B, 16)
                social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
            
            #loss = 1.0*dist_loss + 1.0*diff_loss + 5.0*loss_geo
            loss = 4.0*dist_loss + 0.0*distall_loss + 0.5*diff_loss + 10.0*loss_geo + 100.0*social_loss + 10.0*personal_loss + 0.001*dist_temp_loss
            """
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item() 
            #print(dist_loss, diff_loss, loss_cpu)            
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation_gps(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    actions,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
            
                """
                visualize_exaug_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    batch_3d_point,
                    goal_pos,
                    local_yaw,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    norm*ped_past_c.cpu(),
                    #ped_list_raw,
                    #ped_list_no_trans,
                    norm*est_ped_traj.cpu(),
                    norm*est_ped_traj_zeros.cpu(),
                    norm*robot_past_c.cpu(),
                    last_poses.cpu(),
                    rsize.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
                """          

def evaluate_il_dist_gnm_gps_map2_lan(
    eval_type: str,
    ema_model: EMAModel,
    #optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    #noise_scheduler: DDPMScheduler,
    #goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    sacson: bool,
    #model_depth,
    #model_pedtraj,
    #device2,         
    len_traj_pred: int,    
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,    
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model  
    ema_model.eval()
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    total_loss_logger = Logger("total_loss", "test", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)    
    
    loggers = {
        "total_loss": total_loss_logger,
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,                     
    }
    """
    D = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/D.npy', mmap_mode='r'))
    K = np.array(np.load('/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/vint_train/training/fisheye_calibration/K.npy', mmap_mode='r'))

    mask_360 = np.loadtxt(open("/media/noriaki/Noriaki_Data/learning-language-navigation_SACSoN/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)    
    mask_360_resize = np.repeat(np.expand_dims(cv2.resize(mask_360, (832, 128)), 0), 3, 0).astype(np.float32)
    mask_360_torch = torch.from_numpy(mask_360_resize[:,:,0:416]).unsqueeze(0).to(device2)
    """
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:      
        for i, data in enumerate(tepoch):
            """
            (
                obs_image, 
                goal_image,
                current_image,                
                actions,
                distance,
                goal_pos,
                local_goal_mat,
                local_yaw,                   
                _,
                _,
                #_, 
                id_num,
                action_mask,
                ped_list,
                ped_list_raw,
                ped_list_no_trans,
                robot_list,                
            ) = data        
            """  
            (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    goal_yaw,
                    dataset_index,
                    action_mask,
                    #_,
                    #_,
                ) = data                   
            #at different GPUs
            if psutil.virtual_memory().percent > 90.0:
                print("RAM usage (%)", psutil.virtual_memory().percent)
                break   
            """
            current_image_depth = (current_image.to(device2))*mask_360_torch
            B, _, H, W = current_image_depth.size()             
            
            with torch.no_grad():
                proj_3d, outputs = model_depth.forward(current_image_depth) #for depth360   
                
            #batch_3d_point = proj_3d.to(device)
            batch_3d_point_cpu = proj_3d.cpu()
            batch_3d_point = batch_3d_point_cpu.to(device) 
            """                      
            B, _, H, W = obs_image.size()  

            current_map_image = torch.zeros(B, 3, 96, 96)
            goal_map_image = torch.zeros(B, 3, 96, 96)
            
            local_yaw = torch.ones(B)
              
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            viz_obs_image_past = TF.resize(obs_images[0], VISUALIZATION_IMAGE_SIZE[::-1])   
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image_map = obs_images[0].to(device)        
            obs_image = torch.cat(obs_images, dim=1)
            
            map_images = torch.cat((current_map_image.to(device), goal_map_image.to(device), obs_image_map), axis=1)                
            actions = actions.to(device)
            #print("actions", actions.size())
            
            action_mask = action_mask.to(device)
            
            batch_goal_pos = goal_pos.to(device)
            far_goal_mask = (torch.abs(batch_goal_pos[:,0]) < 5.0) * (torch.abs(batch_goal_pos[:,1]) < 5.0)

            distance = distance.float().to(device)
            goal_mask = (distance > 0.1) * far_goal_mask
            goal_pose_gps = torch.cat((goal_pos, torch.cos(goal_yaw).unsqueeze(1), torch.sin(goal_yaw).unsqueeze(1)), axis=1).to(device)
            # Get distance label
            #distance = distance.float().to(device)
            #goal_mask = distance > 0.1
                        
            #print(goal_mask)
            """
            for ig in range(B):
                if not goal_mask[ig]:
                    distance[ig] = 20
                    igr = random.randint(0, B-1) 
                    while ig == igr:
                        igr = random.randint(0, B-1) 
                    goal_image[ig] = goal_image[igr]
            """
            goal_mask_gnm = []
            for idg in range(B):
                if distance[idg] <= 20:
                    goal_mask_gnm.append(random.randint(4,6))
                else:
                    goal_mask_gnm.append(random.randint(4,5))
            goal_mask_sub = torch.tensor(goal_mask_gnm).to(device)                 
            
            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_goal_pos = goal_pos.to(device)
            goal_image = transform(goal_image).to(device)

            #model calculate 
            #goal_mask_f = torch.randint(0,3,(B,)) #Frodobot: Goal pose, or Satellite or (Goal pose + Satellite)                   
            #goal_mask_sub = torch.zeros(B).to(device) #GNM: Goal pose only
            #goal_mask = torch.cat((goal_mask_f, goal_mask_sub), axis=0)
            
            with torch.no_grad():
                action_pred, dist_pred, _ = ema_model(obs_image, goal_pose_gps, map_images, goal_image, goal_mask_sub)
                #linear_vel, angular_vel, dist_temp = ema_model(obs_image, goal_image, rsize)

            #print("distance", distance)
            #print("dist_pred", dist_pred)
            #dist_loss = F.mse_loss(dist_pred.squeeze(-1), distance.float())
            #print(dist_loss)
            """
            losses = _compute_losses_gps(
                action_label=actions,
                action_pred=action_pred,
                learn_angle=True,
                action_mask=action_mask,
            )
            """
            
            losses = _compute_losses(
                dist_label=distance,
                action_label=actions,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=0.5,
                learn_angle=True,
                action_mask=action_mask,
            )
            
            """                
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((pz_ref.unsqueeze(1), -px_ref.unsqueeze(1)), axis=1) #from camera coordinate to robot local coordinate
       
            mat_1 = torch.cat((torch.cos(-ry_ref).unsqueeze(1), -torch.sin(-ry_ref).unsqueeze(1), 2.0*pz_ref.unsqueeze(1)), axis=1)
            mat_2 = torch.cat((torch.sin(-ry_ref).unsqueeze(1), torch.cos(-ry_ref).unsqueeze(1), -2.0*px_ref.unsqueeze(1)), axis=1)
            mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
            last_pose_mat = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)            
            
            robot_traj_list = []
            for ip in range(len(px_ref_list)):
                mat_1 = torch.cat((torch.cos(-ry_ref_list[ip]).unsqueeze(1), -torch.sin(-ry_ref_list[ip]).unsqueeze(1), 2.0 * pz_ref_list[ip].unsqueeze(1)), axis=1)
                mat_2 = torch.cat((torch.sin(-ry_ref_list[ip]).unsqueeze(1), torch.cos(-ry_ref_list[ip]).unsqueeze(1), -2.0 * px_ref_list[ip].unsqueeze(1)), axis=1)
                mat_3 = torch.cat((torch.zeros(B,1), torch.zeros(B,1), torch.ones(B,1)), axis=1).to(device)   
                mat_combine = torch.cat((mat_1.unsqueeze(1), mat_2.unsqueeze(1), mat_3.unsqueeze(1)), axis=1)
                robot_traj_list.append(mat_combine.unsqueeze(1))
            robot_traj_vec = torch.cat(robot_traj_list, axis=1)

            local_goal_mat[:, 0,2] *= 2.0                    
            local_goal_mat[:, 1,2] *= 2.0  
            local_goal_vec = local_goal_mat.unsqueeze(1).repeat(1,8,1,1)

            dist_loss = nn.functional.mse_loss(last_pose_mat[goal_mask], local_goal_mat.to(device)[goal_mask])                                    
            #dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos) 
            distall_loss = nn.functional.mse_loss(robot_traj_vec[goal_mask], local_goal_vec.to(device)[goal_mask])                     
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1][goal_mask], linear_vel[:,1:][goal_mask]) + nn.functional.mse_loss(angular_vel[:,:-1][goal_mask], angular_vel[:,1:][goal_mask]) 

            # Get distance label
            distance = distance.float().to(device)

            # Predict distance         
            dist_temp_loss = F.mse_loss(dist_temp.squeeze(-1), distance)

            PC3D = []
            for j in range(len_traj_pred):
                px_ref = px_ref_list[j]
                pz_ref = pz_ref_list[j]
                ry_ref = ry_ref_list[j]                

                Tod = torch.zeros((B, 4, 4)).to(device)
                Tod[:, 0, 0] = torch.cos(ry_ref)
                Tod[:, 0, 2] = torch.sin(ry_ref)
                Tod[:, 1, 1] = 1.0
                Tod[:, 2, 0] = -torch.sin(ry_ref)
                Tod[:, 2, 2] = torch.cos(ry_ref)
                Tod[:, 0, 3] = px_ref
                Tod[:, 2, 3] = pz_ref
                Tod[:, 3, 3] = 1.0

                Ttrans = torch.inverse(Tod)[:, :3, :]               
                batch_3d_point_x = torch.cat((batch_3d_point.view(B, 3, -1), torch.ones(B,1,416*128).to(device)), axis=1)
                cam_points_trans = torch.matmul(Ttrans, batch_3d_point_x).view(B, 3, 128, 416)
                PC3D.append(cam_points_trans.unsqueeze(1))                                                  
            
            PC3D_cat = torch.cat(PC3D, axis=1)                    
  
            loss_geo = geometry_criterion(PC3D_cat[goal_mask], rsize[:,:,0][goal_mask], len_traj_pred, device)
            norm = 10.0
            if sacson:
                ped_past = torch.cat((-torch.flip(ped_list[:,:,1], dims=[1]), torch.flip(ped_list[:,:,0], dims=[1])), axis=1).to(device)  
                robot_past = torch.cat((-torch.flip(robot_list[:,:,1], dims=[1]), torch.flip(robot_list[:,:,0], dims=[1])), axis=1).to(device)
                
                robot_future_x = []
                robot_future_z = []
                for ir in range(len(px_ref_list)):
                    robot_future_x.append(px_ref_list[ir].unsqueeze(1))
                    robot_future_z.append(pz_ref_list[ir].unsqueeze(1))                    
                robot_future = torch.cat((torch.cat(robot_future_x, axis=1), torch.cat(robot_future_z, axis=1)), axis=1).to(device)  
                            
                ped_past_c = torch.clamp(ped_past, min=-10.0, max=10.0)/norm
                robot_past_c = torch.clamp(robot_past, min=-10.0, max=10.0)/norm
                robot_future_c = torch.clamp(robot_future, min=-10.0, max=10.0)/norm
                
                flag_ped = (ped_past_c[:,0] != 0.0) & (ped_past_c[:,8] != 0.0)
                
                robot_past_c = 0.333*robot_past_c #frodbot is much faster than Vizbot.
                delta_est_ped_traj = model_pedtraj(ped_past_c, robot_past_c, robot_future_c)
                delta_est_ped_traj_zero = model_pedtraj(ped_past_c, robot_past_c, 0.0*robot_future_c)

                traj_x = torch.cumsum(delta_est_ped_traj[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_y = torch.cumsum(delta_est_ped_traj[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj = torch.clamp(torch.cat((traj_x, traj_y), axis=1), min=-10.0, max=10.0)

                delta_est_ped_traj_zero_detach = delta_est_ped_traj_zero.detach()
                traj_xz = torch.cumsum(delta_est_ped_traj_zero_detach[:,0:8]/norm, dim=1) + ped_past_c[:,0:1].repeat(1,8)
                traj_yz = torch.cumsum(delta_est_ped_traj_zero_detach[:,8:16]/norm, dim=1) + ped_past_c[:,8:9].repeat(1,8)           
                est_ped_traj_zeros = torch.clamp(torch.cat((traj_xz, traj_yz), axis=1), min=-10.0, max=10.0)
                
                social_loss = nn.functional.mse_loss(est_ped_traj[flag_ped*goal_mask], est_ped_traj_zeros[flag_ped*goal_mask])
                
                max_pl = rsize.squeeze(1).repeat(1,8) + 0.5
                min_pl = rsize.squeeze(1).repeat(1,8) * 0.0
                personal_loss = nn.functional.mse_loss(max_pl[flag_ped*goal_mask], torch.clamp(torch.sqrt((norm*robot_future_c[flag_ped*goal_mask][:,0:8] - norm*est_ped_traj[flag_ped*goal_mask][:,0:8])**2 + (norm*robot_future_c[flag_ped*goal_mask][:,8:16] - norm*est_ped_traj[flag_ped*goal_mask][:,8:16])**2), min=min_pl[flag_ped*goal_mask], max= max_pl[flag_ped*goal_mask]))
            else:
                est_ped_traj = torch.zeros(B, 16)
                est_ped_traj_zeros = torch.zeros(B, 16)      
                ped_past_c = torch.zeros(B, 16)
                robot_past_c = torch.zeros(B, 16)
                social_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
                personal_loss = nn.functional.mse_loss(est_ped_traj, est_ped_traj_zeros)
            
            #loss = 1.0*dist_loss + 1.0*diff_loss + 5.0*loss_geo
            loss = 4.0*dist_loss + 0.0*distall_loss + 0.5*diff_loss + 10.0*loss_geo + 100.0*social_loss + 10.0*personal_loss + 0.001*dist_temp_loss
            """
            
            # Logging            
            loss_cpu = losses["total_loss"].item()
            dist_loss_cpu = losses["dist_loss"].item()
            action_loss_cpu = losses["action_loss"].item() 
            #print(dist_loss, diff_loss, loss_cpu)            
            
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss_cpu})
            wandb.log({"action_loss": action_loss_cpu})                  
                        
            if i % print_log_freq == 0:
                losses = {}
                losses['total_loss'] = loss_cpu
                losses['dist_loss'] = dist_loss_cpu
                losses['action_loss'] = action_loss_cpu   
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_il_estimation_gps(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    goal_pos,
                    local_yaw,
                    action_pred,
                    actions,
                    "test",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
            
                """
                visualize_exaug_estimation(
                    viz_obs_image, 
                    viz_obs_image_past,                     
                    viz_goal_image,
                    batch_3d_point,
                    goal_pos,
                    local_yaw,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    norm*ped_past_c.cpu(),
                    #ped_list_raw,
                    #ped_list_no_trans,
                    norm*est_ped_traj.cpu(),
                    norm*est_ped_traj_zeros.cpu(),
                    norm*robot_past_c.cpu(),
                    last_poses.cpu(),
                    rsize.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                    )
                """      

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }

def supervision_from_nomad(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    it_num: int,    
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    
    #wandb_list = []
    pred_horizon = 8
    action_dim = 2
    
    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    gc_actions_torch_list = []    
    gc_actions_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        gc_actions_torch_list.append(model_output_dict['gc_actions'])        
    gc_actions_torch_list = torch.concat(gc_actions_torch_list, axis=0)    
    gc_actions_torch_list = torch.split(gc_actions_torch_list, num_samples, dim=0)    
    
    select_traj_list = []
    for i in range(num_images_log):
        gc_actions_torch = gc_actions_torch_list[i]
        gc_actions_torch_cat = torch.concat(torch.split(gc_actions_torch, 1, dim=1), axis=0).squeeze(1)  
        
        batch_goal_pos_i = torch.tensor([batch_goal_pos[i][1], -batch_goal_pos[i][0]])    
        device = gc_actions_torch_cat.get_device()
        
        batch_goal_pos_repeat = batch_goal_pos_i.unsqueeze(0).repeat(num_samples*8, 1).to(device)
        traj_id_all = torch.argmin(torch.sum((batch_goal_pos_repeat - gc_actions_torch_cat)**2, axis=1))
        traj_id = traj_id_all % num_samples
        select_traj_list.append(gc_actions_torch[traj_id:traj_id+1])
    return torch.concat(select_traj_list, axis=0)

def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)

def visualize_lelan_eval(
    batch_viz_obs_images_lan: torch.Tensor,
    batch_viz_goal_images_lan: torch.Tensor,
    batch_viz_goal_images_image: torch.Tensor,
    batch_viz_goal_images_fix: torch.Tensor,      
    goal_pos_lan: torch.Tensor,
    batch_obj_inst_lan: torch.Tensor,
    traj_mbra: torch.Tensor,
    traj_nomad: torch.Tensor,
    traj_est: torch.Tensor,
    mask_number: torch.Tensor,    
    distance: torch.Tensor,   
    project_folder: str,
    eval_type: str,    
    epoch: int,
    num_images_log: int = 30,    
    use_wandb: bool = True,      
):
    """Plot samples from the exploration model."""

    #print(project_folder, eval_type)
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images_lan.shape[0], batch_viz_goal_images_lan.shape[0])    
    #metric_waypoint_spacing = 0.25 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        #ax_past = fig.add_subplot(gs[1:2, 0:1])        
        ax_goal_i = fig.add_subplot(gs[1:2, 1:2])
        ax_goal_f = fig.add_subplot(gs[1:2, 2:3])

        obs_image = to_numpy(batch_viz_obs_images_lan[i][-3:])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images_lan[i])
        goal_image = np.moveaxis(goal_image, 0, -1)    
        
        goal_image_image = to_numpy(batch_viz_goal_images_image[i])
        goal_image_fix = to_numpy(batch_viz_goal_images_fix[i])   
        goal_image_image = np.moveaxis(goal_image_image, 0, -1)
        goal_image_fix = np.moveaxis(goal_image_fix, 0, -1)  
                            
        ax_ob.imshow((obs_image).astype(np.uint8))               
        ax_goal.imshow((goal_image).astype(np.uint8))   
        ax_goal_i.imshow((255*goal_image_image).astype(np.uint8))               
        ax_goal_f.imshow((255*goal_image_fix).astype(np.uint8))   
                                            
        #xgt = to_numpy(goal_pos_lan[i,1])
        #ygt = -to_numpy(goal_pos_lan[i,0])
        xgt = to_numpy(goal_pos_lan[i,0])
        ygt = to_numpy(goal_pos_lan[i,1])
                
        mask_type = mask_number[i]        
        x_mbra = traj_mbra[i, :, 0].detach().cpu().numpy()
        y_mbra = traj_mbra[i, :, 1].detach().cpu().numpy()
        x_nomad = traj_nomad[i, :, 0].detach().cpu().numpy()
        y_nomad = traj_nomad[i, :, 1].detach().cpu().numpy()
        x_est = traj_est[i, :, 0].detach().cpu().numpy()
        y_est = traj_est[i, :, 1].detach().cpu().numpy()
                                
        ax_graph.plot(-y_mbra, x_mbra, marker = 'o', color='blue', label="mbra")        
        ax_graph.plot(-y_nomad, x_nomad, marker = 'o', color='red', label="nomad")               
        ax_graph.plot(-y_est, x_est, marker = 'o', color='magenta', label="est")               
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')   
                     
        #ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')  
        #ax_graph.annotate("X:" + str(xgt) + "Y:" + str(ygt), xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')         
        #ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        if mask_type == 0:
            ax_graph.annotate("satellite only " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 1:
            ax_graph.annotate("pose and satellite " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')           
        elif mask_type == 2:
            ax_graph.annotate("satellite and image " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
        elif mask_type == 3:
            ax_graph.annotate("all " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 4:
            ax_graph.annotate("pose only " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
        elif mask_type == 5:
            ax_graph.annotate("pose and image " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 6:
            ax_graph.annotate("image only " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 7:
            ax_graph.annotate("language only " + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
            ax_graph.annotate(batch_obj_inst_lan[i], xy = (-0.0, 0.0), xytext = (-20, -40),textcoords = 'offset points')   
        elif mask_type == 8:
            ax_graph.annotate("pose and language" + str(distance[i].cpu().numpy()), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
            ax_graph.annotate(batch_obj_inst_lan[i], xy = (-0.0, 0.0), xytext = (-20, -40),textcoords = 'offset points')               
                                                                    
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-5.0, 5.0)
        ax_graph.set_ylim(-1.0, 4.0)
        ax_graph.legend(loc='best')                  
        #ax_ob.set_title(f"observation")
        ax_goal.set_title(f"cropped goal image")
        ax_goal_i.set_title(f"selected goal image")
        ax_goal_f.set_title(f"fixed goal image")                
        #ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_lelan_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def visualize_exaug_delay_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    batch_3d_point: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    ped_list: torch.Tensor, 
    est_ped_traj: torch.Tensor,
    est_ped_traj_zeros: torch.Tensor,   
    robot_list: torch.Tensor,
    last_poses: torch.Tensor,
    rsize: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0], last_poses.shape[0])    
    batch_linear_vel = linear_vel[:num_images_log]
    batch_angular_vel = angular_vel[:num_images_log]
    
    px_list, pz_list, ry_list = robot_pos_model_fix(batch_linear_vel, batch_angular_vel)
    
    px_list_a = []
    pz_list_a = []
    for px_v in px_list:
        px_list_a.append(px_v.unsqueeze(1))
    for pz_v in pz_list:
        pz_list_a.append(pz_v.unsqueeze(1))        
    batch_px_list = torch.cat(px_list_a, axis=1)
    batch_pz_list = torch.cat(pz_list_a, axis=1)
    last_yaw = to_numpy(ry_list[-1])
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:1, 0:1])
        ax_graph2 = fig.add_subplot(gs[1:2, 0:1])        
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_depth1 = fig.add_subplot(gs[1:2, 1:2])
                            
        x_seq_old = to_numpy(batch_px_list[i, 0:6])
        z_seq_old = to_numpy(batch_pz_list[i, 0:6])
        x_seq = to_numpy(batch_px_list[i, 6:14])
        z_seq = to_numpy(batch_pz_list[i, 6:14])
                        
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14
        label_action = last_yaw[i] * 180 / 3.14
        
        hup = 48#150
        hdown = 64#190
        bias = 10        
        #bias_x = 10
        batch_3d_point_flatten = to_numpy(batch_3d_point[i,:,hup:hdown,bias:416-bias]).reshape(3,-1)        
        batch_3d_point = batch_3d_point.cpu()
        mask1 = (batch_3d_point[i,1:2,:,:].cpu() < 0.0*torch.ones((1, 128, 416)))
        mask1_x = (batch_3d_point[i,1:2,:,:].cpu() < 0.0*torch.ones((1, 128, 416)))        
        mask2 = (batch_3d_point[i,1:2,:,:].cpu() > -0.3*torch.ones((1, 128, 416)))    
        mask2_x = (batch_3d_point[i,1:2,:,:].cpu() > -0.3*torch.ones((1, 128, 416)))                    
        mask = torch.logical_and(mask1, mask2)[:,:,bias:416-bias]

        ped_x = ped_list[i,0:8].detach().numpy()                      
        ped_y = ped_list[i,8:16].detach().numpy()  
         
        robot_x = robot_list[i,0:8].detach().numpy()                      
        robot_y = robot_list[i,8:16].detach().numpy()          
                      
        pedest_x = est_ped_traj[i,0:8].detach().numpy()                      
        pedest_z = est_ped_traj[i,8:16].detach().numpy()  
        pedest_zero_x = est_ped_traj_zeros[i,0:8].detach().numpy()
        pedest_zero_z = est_ped_traj_zeros[i,8:16].detach().numpy() 
                                           
        xenv = to_numpy(batch_3d_point[i,0:1,:,bias:416-bias][mask]).reshape(-1)
        zenv = to_numpy(batch_3d_point[i,2:3,:,bias:416-bias][mask]).reshape(-1)  

        ax_graph.plot(x_seq_old, z_seq_old, marker = 's', color='cyan')                
        ax_graph.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph.plot(ped_x, ped_y, marker = 'o', color='magenta')
        ax_graph.plot(ped_x[0], ped_y[0], marker = 's', color='red')   
        ax_graph.plot(pedest_zero_x, pedest_zero_z, marker = 'o', color='cyan')  
        ax_graph.plot(pedest_x, pedest_z, marker = 'o', color='orange')              
          
        ax_graph.plot(robot_x, robot_y, marker = 'o', color='green')
        ax_graph.plot(robot_x[0], robot_y[0], marker = 's', color='green')                          
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')
        ax_graph.plot(-yest, xest, marker = '+', color='green')                   
        ax_graph.scatter(xenv, zenv, marker = '.', color='black')     
        ax_graph.annotate(str(label)+' degrees', xy = (-ygt, xgt), xytext = (-20, 20),textcoords = 'offset points')

        ax_graph2.plot(x_seq_old, z_seq_old, marker = 's', color='cyan')            
        ax_graph2.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph2.plot(ped_x, ped_y, marker = 'o', color='magenta')
        ax_graph2.plot(ped_x[0], ped_y[0], marker = 's', color='red')           
        ax_graph2.plot(pedest_zero_x, pedest_zero_z, marker = 'o', color='cyan')  
        ax_graph2.plot(pedest_x, pedest_z, marker = 'o', color='orange')          
        ax_graph2.plot(robot_x, robot_y, marker = 'o', color='green')
        ax_graph2.plot(robot_x[0], robot_y[0], marker = 's', color='green')                                         
        ax_graph2.plot(-ygt, xgt, marker = '*', color='red')
        ax_graph2.plot(-yest, xest, marker = '+', color='green')          
        ax_graph2.scatter(xenv, zenv, marker = '.', color='black')
        ax_graph2.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph2.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')
        ang_vel = (180.0*0.333*torch.sum(angular_vel[i,:])/3.1415).cpu().detach().numpy()
        ax_graph2.annotate(str(ang_vel)+' degrees (Vel)', xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')
                                                        
        for j in range(8):
            circle = plt.Circle((x_seq[j], z_seq[j]), to_numpy(rsize)[i,0,0], color='black', fill=False)
            ax_graph2.add_patch(circle)          
        
        ax_past.plot(linear_vel[i,:].cpu().detach().numpy(), marker = 'o', color='red')
        ax_past.plot(angular_vel[i,:].cpu().detach().numpy(), marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)        
        obs_image_past = to_numpy(batch_viz_obs_images_past[i])
        obs_image_past = np.moveaxis(obs_image_past, 0, -1)             
        goal_image = to_numpy(batch_viz_goal_images[i])
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))              
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
            
        ax_depth1.imshow(to_numpy(batch_3d_point[i,2,:,:]).astype(np.uint8), cmap='jet', interpolation='nearest')
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-10, 10)
        ax_graph.set_ylim(-1, 10)       
        ax_graph2.set_title(f"est. trajectory")
        ax_graph2.set_xlim(-5.0, 5.0)
        ax_graph2.set_ylim(-5.0, 5.0)            
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))            
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)            
            
        plt.close(fig)

def visualize_il_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    #print(project_folder, eval_type)
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 0.3 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_depth1 = fig.add_subplot(gs[1:2, 1:2])
                            
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        
                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue')                       
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
                        
        # set title
        ax_graph.set_title(f"est. trajectory")               
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)       
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def visualize_il_estimation_gps(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    action: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 1.0 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_yaw = fig.add_subplot(gs[1:2, 1:2])
                            
        xgt = to_numpy(obj_poses[i,0])*metric_waypoint_spacing
        ygt = to_numpy(obj_poses[i,1])*metric_waypoint_spacing
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        yaw_seq = torch.atan2(action_pred[i, :, 3], action_pred[i, :, 2]).detach().cpu().numpy() * 180 / 3.14
        yaw_label = torch.atan2(action[i, :, 3], action[i, :, 2]).detach().cpu().numpy() * 180 / 3.14
        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        x_label = action[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_label = action[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing        
                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue')        
        ax_graph.plot(-y_label, x_label, marker = 'o', color='red')                            
        ax_graph.plot(-ygt, xgt, marker = '*', markersize=15, color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
        ax_past.plot(x_label, marker = 's', color='red')
        ax_past.plot(y_label, marker = 's', color='blue')
                        
        ax_yaw.plot(yaw_seq, marker = 'o', color='blue')
        ax_yaw.plot(yaw_label, marker = 's', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
                        
        # set title
        ax_graph.set_title(f"est. trajectory")                
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def visualize_il2_estimation_map(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    batch_viz_cur_map: torch.Tensor,
    batch_viz_goal_map: torch.Tensor,    
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    action_label: torch.Tensor,    
    action_origin: torch.Tensor,  
    mask_number: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 0.25 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:1, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 0:1])        
        ax_curmap = fig.add_subplot(gs[1:2, 1:2])
        ax_goalmap = fig.add_subplot(gs[1:2, 2:3])
                            
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        mask_type = mask_number[i]        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        
        x_seq_l = action_label[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        y_seq_l = action_label[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        x_seq_o = action_origin[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq_o = action_origin[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
                                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue', label="est")        
        ax_graph.plot(-y_seq_l, x_seq_l, marker = 'o', color='red', label="label")               
        ax_graph.plot(-y_seq_o, x_seq_o, marker = 'o', color='magenta', label="original label")               
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')  
        ax_graph.annotate("X:" + str(xgt) + "Y:" + str(ygt), xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')         
        if mask_type == 0:
            ax_graph.annotate("pose only", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 1:
            ax_graph.annotate("satellite only", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')           
        else:
            ax_graph.annotate("pose and satellite", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
                    
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        

        map_image_cur = to_numpy(batch_viz_cur_map[i])
        map_image_cur = np.moveaxis(map_image_cur, 0, -1)                    
        map_image_goal = to_numpy(batch_viz_goal_map[i])
        map_image_goal = np.moveaxis(map_image_goal, 0, -1)        
        ax_curmap.imshow((255.0*map_image_cur).astype(np.uint8))               
        ax_goalmap.imshow((255.0*map_image_goal).astype(np.uint8))     
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-2.0, 2.0)
        ax_graph.set_ylim(-1.0, 3.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def visualize_il2_estimation_map2(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    batch_viz_cur_map: torch.Tensor,
    batch_viz_goal_map: torch.Tensor,    
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    action_label: torch.Tensor,    
    action_origin: torch.Tensor,  
    mask_number: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    #print(project_folder, eval_type)
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 0.25 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:1, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 0:1])        
        ax_curmap = fig.add_subplot(gs[1:2, 1:2])
        ax_goalmap = fig.add_subplot(gs[1:2, 2:3])
                            
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        mask_type = mask_number[i]        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        
        x_seq_l = action_label[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        y_seq_l = action_label[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        x_seq_o = action_origin[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq_o = action_origin[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
                                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue', label="est")        
        ax_graph.plot(-y_seq_l, x_seq_l, marker = 'o', color='red', label="label")               
        ax_graph.plot(-y_seq_o, x_seq_o, marker = 'o', color='magenta', label="original label")               
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')  
        ax_graph.annotate("X:" + str(xgt) + "Y:" + str(ygt), xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')         
        #ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        if mask_type == 0:
            ax_graph.annotate("satellite only", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 1:
            ax_graph.annotate("pose and satellite", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')           
        elif mask_type == 2:
            ax_graph.annotate("satellite and image", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
        elif mask_type == 3:
            ax_graph.annotate("all", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 4:
            ax_graph.annotate("pose only", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
        elif mask_type == 5:
            ax_graph.annotate("pose and image", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 6:
            ax_graph.annotate("image only", xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
                                                        
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        

        map_image_cur = to_numpy(batch_viz_cur_map[i])
        map_image_cur = np.moveaxis(map_image_cur, 0, -1)                    
        map_image_goal = to_numpy(batch_viz_goal_map[i])
        map_image_goal = np.moveaxis(map_image_goal, 0, -1)        
        ax_curmap.imshow((255.0*map_image_cur).astype(np.uint8))               
        ax_goalmap.imshow((255.0*map_image_goal).astype(np.uint8))     
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-2.0, 2.0)
        ax_graph.set_ylim(-1.0, 3.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)


def visualize_il2_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_obs_images_past: torch.Tensor,    
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    local_yaw: torch.Tensor,
    action_pred: torch.Tensor,
    action_label: torch.Tensor,    
    action_origin: torch.Tensor,  
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0])    
    metric_waypoint_spacing = 0.25 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_past = fig.add_subplot(gs[1:2, 2:3])        
        ax_depth1 = fig.add_subplot(gs[1:2, 1:2])
                            
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])
        
        if (local_yaw[i].item()) % (2.0*3.14) > 3.14:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14) - 2.0*3.14
        else:
            ang_yaw = (local_yaw[i].item()) % (2.0*3.14)
        label = ang_yaw * 180 / 3.14        
        label_action = torch.atan(action_pred[i, -1, 3]/action_pred[i, -1, 2]) * 180 / 3.14
        
        x_seq = action_pred[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq = action_pred[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
        
        x_seq_l = action_label[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        y_seq_l = action_label[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing*0.5
        x_seq_o = action_origin[i, :, 0].detach().cpu().numpy()*metric_waypoint_spacing
        y_seq_o = action_origin[i, :, 1].detach().cpu().numpy()*metric_waypoint_spacing
                                
        ax_graph.plot(-y_seq, x_seq, marker = 'o', color='blue', label="est")        
        ax_graph.plot(-y_seq_l, x_seq_l, marker = 'o', color='red', label="label")               
        ax_graph.plot(-y_seq_o, x_seq_o, marker = 'o', color='magenta', label="original label")               
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')                
        ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')        
        ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        
        ax_past.plot(x_seq, marker = 'o', color='red')
        ax_past.plot(y_seq, marker = 'o', color='blue')
                                              
        obs_image = to_numpy(batch_viz_obs_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images[i])
        goal_image = np.moveaxis(goal_image, 0, -1)        
        ax_ob.imshow((255.0*obs_image).astype(np.uint8))               
        ax_goal.imshow((255.0*goal_image).astype(np.uint8))        
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-2.0, 2.0)
        ax_graph.set_ylim(-1.0, 3.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal image")
        ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_ped_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)


def visualize_lelan_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    obj_inst: torch.Tensor,
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    last_poses: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0], last_poses.shape[0])    
    batch_linear_vel = linear_vel[:num_images_log]
    batch_angular_vel = angular_vel[:num_images_log]
    
    px_list, pz_list, ry_list = robot_pos_model_fix(batch_linear_vel, batch_angular_vel)
    
    px_list_a = []
    pz_list_a = []
    for px_v in px_list:
        px_list_a.append(px_v.unsqueeze(1))
    for pz_v in pz_list:
        pz_list_a.append(pz_v.unsqueeze(1))        
    batch_px_list = torch.cat(px_list_a, axis=1)
    batch_pz_list = torch.cat(pz_list_a, axis=1)
    
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_inst = fig.add_subplot(gs[1:2, 1:3])
                    
        x_seq = to_numpy(batch_px_list[i])
        z_seq = to_numpy(batch_pz_list[i])
                
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        ax_graph.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph.plot(xgt, ygt, marker = '*', color='red')
        ax_graph.plot(xest, yest, marker = '+', color='green')
                
        obs_image = to_numpy(batch_viz_obs_images[i])
        prompt = obj_inst[i]
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax_ob.imshow(obs_image)
        ax_goal.imshow(goal_image)
        ax_inst.text(0, 0, prompt, fontsize = 12, color = 'black')
        ax_inst.axis('off')
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"cropped goal image")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
            
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)       
        
def visualize_BDD_eval(
    batch_viz_obs_images_lan: torch.Tensor,
    batch_viz_goal_images_lan: torch.Tensor,
    batch_viz_goal_images_lan_8: torch.Tensor,    
    #batch_viz_goal_images_image: torch.Tensor,
    #batch_viz_goal_images_fix: torch.Tensor,      
    goal_pos_lan: torch.Tensor,
    #batch_obj_inst_lan: torch.Tensor,
    traj_mbra: torch.Tensor,
    #traj_nomad: torch.Tensor,
    traj_est: torch.Tensor,
    mask_number: torch.Tensor,    
    distance: torch.Tensor,   
    project_folder: str,
    eval_type: str,    
    epoch: int,
    num_images_log: int = 30,    
    use_wandb: bool = True,      
):
    """Plot samples from the exploration model."""

    #print(project_folder, eval_type)
    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images_lan.shape[0])    
    #metric_waypoint_spacing = 0.25 #normalization   
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 2:3])
        ax_goal = fig.add_subplot(gs[1:2, 2:3])
        #ax_past = fig.add_subplot(gs[1:2, 0:1])        
        ax_graph_2 = fig.add_subplot(gs[0:1, 1:2])
        ax_goal_8 = fig.add_subplot(gs[1:2, 1:2])
        #ax_goal_f = fig.add_subplot(gs[1:2, 2:3])

        obs_image = to_numpy(batch_viz_obs_images_lan[i][-3:])
        obs_image = np.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(batch_viz_goal_images_lan[i])
        goal_image = np.moveaxis(goal_image, 0, -1)    
        goal_image_8 = to_numpy(batch_viz_goal_images_lan_8[i])
        goal_image_8 = np.moveaxis(goal_image_8, 0, -1)          
        #goal_image_image = to_numpy(batch_viz_goal_images_image[i])
        #goal_image_fix = to_numpy(batch_viz_goal_images_fix[i])   
        #goal_image_image = np.moveaxis(goal_image_image, 0, -1)
        #goal_image_fix = np.moveaxis(goal_image_fix, 0, -1)  
                            
        ax_ob.imshow((obs_image).astype(np.uint8))               
        ax_goal.imshow((goal_image).astype(np.uint8))  
        ax_goal_8.imshow((goal_image_8).astype(np.uint8))           
        #ax_goal_i.imshow((255*goal_image_image).astype(np.uint8))               
        #ax_goal_f.imshow((255*goal_image_fix).astype(np.uint8))   
                                            
        #xgt = to_numpy(goal_pos_lan[i,1])
        #ygt = -to_numpy(goal_pos_lan[i,0])
        xgt = to_numpy(goal_pos_lan[i,0])
        ygt = to_numpy(goal_pos_lan[i,1])  
        
        angle_rad = np.arctan2(to_numpy(goal_pos_lan[i,3]), to_numpy(goal_pos_lan[i,2]))
        angle_deg = np.degrees(angle_rad)
             
        mask_type = mask_number[i]        
        x_mbra = traj_mbra[i, :, 0].detach().cpu().numpy()
        y_mbra = traj_mbra[i, :, 1].detach().cpu().numpy()
        #x_nomad = traj_nomad[i, :, 0].detach().cpu().numpy()
        #y_nomad = traj_nomad[i, :, 1].detach().cpu().numpy()
        x_est = traj_est[i, :, 0].detach().cpu().numpy()
        y_est = traj_est[i, :, 1].detach().cpu().numpy()
                                
        ax_graph.plot(-y_mbra, x_mbra, marker = 'o', color='blue', label="mbra")        
        #ax_graph.plot(-y_nomad, x_nomad, marker = 'o', color='red', label="nomad")               
        ax_graph.plot(-y_est, x_est, marker = 'o', color='magenta', label="est")               
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')   

        ax_graph_2.plot(-y_mbra, x_mbra, marker = 'o', color='blue', label="mbra")                 
        ax_graph_2.plot(-y_est, x_est, marker = 'o', color='magenta', label="est")               
        ax_graph_2.plot(-ygt, xgt, marker = '*', color='red')   
                             
        #ax_graph.annotate(str(label)+' degrees (GT)', xy = (0.0, 0.0), xytext = (-20, 20),textcoords = 'offset points')  
        #ax_graph.annotate("X:" + str(xgt) + "Y:" + str(ygt), xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')         
        #ax_graph.annotate(str(label_action)+' degrees', xy = (-0.0, 0.0), xytext = (-20, 00),textcoords = 'offset points')   
        if mask_type == 0:
            ax_graph.annotate("satellite only " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 1:
            ax_graph.annotate("pose and satellite " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')           
        elif mask_type == 2:
            ax_graph.annotate("satellite and image " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
        elif mask_type == 3:
            ax_graph.annotate("all " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 4:
            ax_graph.annotate("pose only " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')    
        elif mask_type == 5:
            ax_graph.annotate("pose and image " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 6:
            ax_graph.annotate("image only " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
        elif mask_type == 7:
            ax_graph.annotate("language only " + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
            ax_graph.annotate(batch_obj_inst_lan[i], xy = (-0.0, 0.0), xytext = (-20, -40),textcoords = 'offset points')   
        elif mask_type == 8:
            ax_graph.annotate("pose and language" + str(distance[i].cpu().numpy()) + ", " + str(angle_deg), xy = (-0.0, 0.0), xytext = (-20, -20),textcoords = 'offset points')   
            ax_graph.annotate(batch_obj_inst_lan[i], xy = (-0.0, 0.0), xytext = (-20, -40),textcoords = 'offset points')               
                                                                    
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_graph.set_xlim(-5.0, 5.0)
        ax_graph.set_ylim(-1.0, 4.0)
        ax_graph.legend(loc='best')        
        ax_graph_2.set_title(f"est. trajectory")
        ax_graph_2.legend(loc='best')               
                  
        ax_ob.set_title(f"current image")
        ax_goal.set_title(f"goal image")
        ax_goal_8.set_title(f"next image")        
        #ax_goal_i.set_title(f"selected goal image")
        #ax_goal_f.set_title(f"fixed goal image")                
        #ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_BDD_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)
        
        
def visualize_lelan_col_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    obj_inst: torch.Tensor,
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    last_poses: torch.Tensor,
    ref_actions: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0], last_poses.shape[0])    
    batch_linear_vel = linear_vel[:num_images_log]
    batch_angular_vel = angular_vel[:num_images_log]
    
    px_list, pz_list, ry_list = robot_pos_model_fix(batch_linear_vel, batch_angular_vel)
    
    px_list_a = []
    pz_list_a = []
    for px_v in px_list:
        px_list_a.append(px_v.unsqueeze(1))
    for pz_v in pz_list:
        pz_list_a.append(pz_v.unsqueeze(1))        
    batch_px_list = torch.cat(px_list_a, axis=1)
    batch_pz_list = torch.cat(pz_list_a, axis=1)

    wandb_list = []
        
    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_inst = fig.add_subplot(gs[1:2, 1:3])
                    
        x_seq = to_numpy(batch_px_list[i])
        z_seq = to_numpy(batch_pz_list[i])
                
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        x_nomad = to_numpy(ref_actions[i,:,0])
        y_nomad = to_numpy(ref_actions[i,:,1])
        
        ax_graph.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph.plot(-y_nomad, x_nomad, marker = 'o', color='magenta')
        ax_graph.plot(xgt, ygt, marker = '*', color='red')
        ax_graph.plot(xest, yest, marker = '+', color='green')
                
        obs_image = to_numpy(batch_viz_obs_images[i])
        prompt = obj_inst[i]
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax_ob.imshow(obs_image)
        ax_goal.imshow(goal_image)
        ax_inst.text(0, 0, prompt, fontsize = 12, color = 'black')
        ax_inst.axis('off')
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"cropped goal image")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
            
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)           
        
def calculate_fov(K, image_size):
    # Extract focal lengths from the intrinsic matrix
    f_x = K[0, 0]
    f_y = K[1, 1]
    
    # Extract image dimensions
    W, H = image_size
    
    # Calculate horizontal and vertical field of view
    hfov = 2 * np.arctan(W / (2 * f_x)) * (180 / np.pi)  # Convert to degrees
    vfov = 2 * np.arctan(H / (2 * f_y)) * (180 / np.pi)  # Convert to degrees
    
    return hfov, vfov


def train_lan_only_ft(
    model,
    text_encoder,
    optimizer,
    dataloader_lan,
    transform,
    device,
    project_folder,
    epoch,
    print_log_freq: int = 20,
    use_wandb: bool = False,
    freeze_backbone: bool = False,
):
    """
    Language-only fine-tune of OmniVLA-edge (IL_gps_map_mask3_lan2) on `frodo_lan` data.
    Uses the precomputed `nomad_traj_norm` from the dataset as the action label, so NO
    runtime NoMaD / ExAug teacher is required. Only the LeLaN stream is used.
    If freeze_backbone, the (frozen) EfficientNet encoders are kept in eval() so their
    BatchNorm running stats are NOT corrupted by the tiny fine-tune set.
    """
    model.train()
    if freeze_backbone:
        model.obs_encoder.eval()
        model.goal_encoder.eval()
        model.goal_encoder_img.eval()
    text_encoder.eval().to(device)

    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    obj_loss_logger = Logger("obj_loss", "train", window_size=print_log_freq)

    with tqdm.tqdm(dataloader_lan, desc=f"lan-ft ep{epoch}", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images_lan,
                goal_image_lan,          # object crop (unused here; kept for tuple parity)
                cur_large_img_lan,
                goal_pos_lan,
                obj_inst_lan,
                goal_pos_norm_lan,
                goal_image_full_lan,
                goal_image_full_8_lan,   # unused in lan-only path
                distance_lan,
                action_mask_lan,
                nomad_traj_lan,          # (B, 8, 4) precomputed trajectory label
            ) = data

            Blan = obs_images_lan.shape[0]

            # ----- image tensors (mirror the lan handling in the full training loop) -----
            obs_images_lan_list = torch.split(obs_images_lan, 3, dim=1)
            obs_image_lan_map = obs_images_lan_list[-1].to(device)        # raw last frame (for map stack)
            obs_image_lan = torch.cat([transform(x).to(device) for x in obs_images_lan_list], dim=1)

            cur_map = torch.zeros(Blan, 3, 96, 96)
            goal_map = torch.zeros(Blan, 3, 96, 96)
            map_images_lan = torch.cat(
                (transform(cur_map).to(device), transform(goal_map).to(device), obs_image_lan_map), axis=1
            )

            goal_img = transform(goal_image_full_lan).to(device)
            cur_large_img = transform(cur_large_img_lan).to(device)

            # ----- text feature -----
            tokens = clip.tokenize(obj_inst_lan, truncate=True).to(device)
            with torch.no_grad():
                feat_text_lan = text_encoder.encode_text(tokens)

            # ----- goal-pose (gps) token from object pose (forward, left) -----
            goal_pos_lan = goal_pos_lan.to(device)
            dis_obj = torch.sqrt(goal_pos_lan[:, 1:2] ** 2 + goal_pos_lan[:, 0:1] ** 2) + 1e-6
            goal_pose_gps_lan = torch.cat(
                (
                    goal_pos_lan[:, 1:2],
                    -goal_pos_lan[:, 0:1],
                    goal_pos_lan[:, 1:2] / dis_obj,
                    -goal_pos_lan[:, 0:1] / dis_obj,
                ),
                axis=1,
            ).to(device)

            # ----- goal modality mask: 7 = language-only, 8 = language + gps -----
            goal_mask_lan = torch.tensor(
                [random.choice([7, 8]) for _ in range(Blan)], dtype=torch.long
            ).to(device)

            action_pred, dist_pred, mask_number = model(
                obs_image_lan,
                goal_pose_gps_lan,
                map_images_lan,
                goal_img,
                goal_mask_lan,
                feat_text_lan,
                cur_large_img,
            )

            # ----- labels -----
            action_label = nomad_traj_lan.to(device)                     # (B, 8, 4)
            metric_waypoint_spacing = 0.125
            tar_obj_pose = goal_pos_lan / metric_waypoint_spacing
            mask_lan = (goal_mask_lan == 7) | (goal_mask_lan == 8)
            dist_label = distance_lan.float().to(device)
            action_mask = action_mask_lan.float().to(device)

            losses = _compute_losses_lan(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                pose_obj_label=tar_obj_pose[mask_lan],
                pose_obj_pred=action_pred[:, -1, 0:2][mask_lan],
                alpha=0.5,
                learn_angle=True,
                image_solo=False,
                sate_solo=False,
                action_mask=action_mask,
            )

            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            total_loss_logger.log_data(losses["total_loss"].item())
            action_loss_logger.log_data(losses["action_loss"].item())
            obj_loss_logger.log_data(losses["obj_loss"].item())

            if print_log_freq != 0 and i % print_log_freq == 0:
                print(
                    f"(epoch {epoch}) (batch {i}/{len(dataloader_lan) - 1}) "
                    f"total={total_loss_logger.latest():.4f} "
                    f"action={action_loss_logger.latest():.4f} "
                    f"obj={obj_loss_logger.latest():.4f}"
                )
                if use_wandb and wandb is not None:
                    wandb.log(
                        {
                            "total_loss": losses["total_loss"].item(),
                            "action_loss": losses["action_loss"].item(),
                            "obj_loss": losses["obj_loss"].item(),
                        }
                    )


@torch.no_grad()
def evaluate_lan_only_ft(model, base_model, text_encoder, dataloader_lan, transform, device, max_batches=None):
    """
    Held-out evaluation for the language-only fine-tune.
    Returns:
      test_action_loss / test_obj_loss : language-goal (mask 7) quality on the test split.
      base_divergence_imagegoal        : how far the fine-tuned model's IMAGE-GOAL (mask 6)
                                         predictions drifted from the frozen base model on the
                                         SAME inputs -> proxy for "did basic driving break".
                                         ~0 = basic driving preserved; large = regression.
    """
    was_training = model.training
    model.eval()
    text_encoder.eval()
    act_losses, obj_losses, base_divs = [], [], []
    for i, data in enumerate(dataloader_lan):
        if max_batches is not None and i >= max_batches:
            break
        (
            obs_images_lan, goal_image_lan, cur_large_img_lan, goal_pos_lan, obj_inst_lan,
            goal_pos_norm_lan, goal_image_full_lan, goal_image_full_8_lan, distance_lan,
            action_mask_lan, nomad_traj_lan,
        ) = data
        Blan = obs_images_lan.shape[0]

        obs_images_lan_list = torch.split(obs_images_lan, 3, dim=1)
        obs_image_lan_map = obs_images_lan_list[-1].to(device)
        obs_image_lan = torch.cat([transform(x).to(device) for x in obs_images_lan_list], dim=1)
        cur_map = torch.zeros(Blan, 3, 96, 96)
        goal_map = torch.zeros(Blan, 3, 96, 96)
        map_images_lan = torch.cat(
            (transform(cur_map).to(device), transform(goal_map).to(device), obs_image_lan_map), axis=1
        )
        goal_img = transform(goal_image_full_lan).to(device)
        goal_img_future = transform(goal_image_full_8_lan).to(device)
        cur_large_img = transform(cur_large_img_lan).to(device)

        tokens = clip.tokenize(obj_inst_lan, truncate=True).to(device)
        feat_text_lan = text_encoder.encode_text(tokens)

        goal_pos_lan = goal_pos_lan.to(device)
        dis_obj = torch.sqrt(goal_pos_lan[:, 1:2] ** 2 + goal_pos_lan[:, 0:1] ** 2) + 1e-6
        goal_pose_gps_lan = torch.cat(
            (goal_pos_lan[:, 1:2], -goal_pos_lan[:, 0:1],
             goal_pos_lan[:, 1:2] / dis_obj, -goal_pos_lan[:, 0:1] / dis_obj), axis=1
        ).to(device)

        # ---- language-goal quality (mask 7 = obs + language) ----
        gm_lan = torch.full((Blan,), 7, dtype=torch.long, device=device)
        a_pred, d_pred, _ = model(
            obs_image_lan, goal_pose_gps_lan, map_images_lan, goal_img, gm_lan, feat_text_lan, cur_large_img
        )
        act_losses.append(F.mse_loss(a_pred, nomad_traj_lan.to(device)).item())
        obj_losses.append(F.mse_loss(a_pred[:, -1, 0:2], goal_pos_lan / 0.125).item())

        # ---- basic-driving regression: IMAGE goal (mask 6 = obs + image), base vs fine-tuned ----
        if base_model is not None:
            gm_img = torch.full((Blan,), 6, dtype=torch.long, device=device)
            a_ft, _, _ = model(
                obs_image_lan, goal_pose_gps_lan, map_images_lan, goal_img_future, gm_img, feat_text_lan, cur_large_img
            )
            a_base, _, _ = base_model(
                obs_image_lan, goal_pose_gps_lan, map_images_lan, goal_img_future, gm_img, feat_text_lan, cur_large_img
            )
            base_divs.append(torch.mean(torch.norm(a_ft[:, :, 0:2] - a_base[:, :, 0:2], dim=-1)).item())

    if was_training:
        model.train()
    return {
        "test_action_loss": float(np.mean(act_losses)) if act_losses else float("nan"),
        "test_obj_loss": float(np.mean(obj_losses)) if obj_losses else float("nan"),
        "base_divergence_imagegoal": float(np.mean(base_divs)) if base_divs else float("nan"),
    }


@torch.no_grad()
def eval_metrics_lan(model, text_encoder, dataloader_lan, transform, device, goal_mask_value=7, mws=0.12):
    """
    Detailed language-goal (default mask 7) metrics on a loader, for base-vs-finetuned comparison.
      action_mse      : MSE over the full (8,4) trajectory vs the dataset nomad_traj_norm (lower better)
      waypoint_err_m  : mean per-waypoint position error in meters (lower better)
      endpoint_err_m  : final-waypoint position error in meters (lower better)
      object_err_m    : final waypoint vs annotated object pose, meters (lower better)
      heading_cos     : mean cos-sim of predicted vs GT heading channels (higher better)
    """
    was_training = model.training
    model.eval(); text_encoder.eval()
    n = 0
    s_act = s_pos = s_end = s_obj = s_cos = 0.0
    for data in dataloader_lan:
        (obs_images_lan, goal_image_lan, cur_large_img_lan, goal_pos_lan, obj_inst_lan,
         goal_pos_norm_lan, goal_image_full_lan, goal_image_full_8_lan, distance_lan,
         action_mask_lan, nomad_traj_lan) = data
        B = obs_images_lan.shape[0]
        obs_list = torch.split(obs_images_lan, 3, dim=1)
        obs_map = obs_list[-1].to(device)
        obs = torch.cat([transform(x).to(device) for x in obs_list], dim=1)
        cur_map = torch.zeros(B, 3, 96, 96); goal_map = torch.zeros(B, 3, 96, 96)
        map_images = torch.cat((transform(cur_map).to(device), transform(goal_map).to(device), obs_map), axis=1)
        goal_img = transform(goal_image_full_lan).to(device)
        cur_large = transform(cur_large_img_lan).to(device)
        tokens = clip.tokenize(obj_inst_lan, truncate=True).to(device)
        feat = text_encoder.encode_text(tokens)
        gp = goal_pos_lan.to(device)
        dis = torch.sqrt(gp[:, 1:2] ** 2 + gp[:, 0:1] ** 2) + 1e-6
        goal_pose = torch.cat((gp[:, 1:2], -gp[:, 0:1], gp[:, 1:2] / dis, -gp[:, 0:1] / dis), axis=1).to(device)
        gm = torch.full((B,), goal_mask_value, dtype=torch.long, device=device)
        pred, dpred, _ = model(obs, goal_pose, map_images, goal_img, gm, feat, cur_large)
        gt = nomad_traj_lan.to(device)
        s_act += F.mse_loss(pred, gt).item() * B
        pos_err = torch.norm(pred[:, :, 0:2] - gt[:, :, 0:2], dim=-1)   # (B,8) normalized units
        s_pos += pos_err.mean().item() * B
        s_end += pos_err[:, -1].mean().item() * B
        s_obj += torch.norm(pred[:, -1, 0:2] * mws - gp, dim=-1).mean().item() * B
        s_cos += F.cosine_similarity(pred[:, :, 2:], gt[:, :, 2:], dim=-1).mean().item() * B
        n += B
    if was_training:
        model.train()
    if n == 0:
        d = {k: float("nan") for k in ["action_mse", "waypoint_err_m", "endpoint_err_m", "object_err_m", "heading_cos"]}
        d["n"] = 0
        return d
    return {
        "action_mse": s_act / n,
        "waypoint_err_m": (s_pos / n) * mws,
        "endpoint_err_m": (s_end / n) * mws,
        "object_err_m": s_obj / n,
        "heading_cos": s_cos / n,
        "n": n,
    }
