import os
import zarr
import wandb
import argparse
import numpy as np
import yaml
import time
import zarr 
import pdb
import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler


from vint_train.training.train_eval_loop import load_model
from vint_train.data.vint_hf_dataset import ViNTLeRobotDataset, ViNTLeRobotDataset_annotate, ViNTDataset_annotate_10k, ViNTLeRobotDataset_IL2, ViNTLeRobotDataset_IL2_gps, ViNTDataset_IL2_gps_10k, ViNTDataset_10k, ViNTDataset_IL2_10k, ViNTLeRobotDataset_IL2_gps_map, ViNTLeRobotDataset_IL2_gps_map_crop, ViNTLeRobotDataset_IL2_gps_map2_crop, ViNTLeRobotDataset_IL2_gps_map_crop_test, EpisodeSampler_IL, EpisodeSampler_annotate, EpisodeSampler_IL_10k, EpisodeSampler_annotate_10k
from vint_train.models.exaug.exaug import ExAug_dist_delay


os.environ["OMP_NUM_THREADS"] = "30"  # Set number of OpenMP threads
os.environ["MKL_NUM_THREADS"] = "30"  # Set number of MKL threads
torch.set_num_threads(30) 


def main(config):

    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # # Load Model 
    # model = ExAug_dist_delay(       
    #             context_size=config["context_size"],
    #             len_traj_pred=config["len_traj_pred"],
    #             learn_angle=config["learn_angle"],
    #             obs_encoder=config["obs_encoder"],
    #             obs_encoding_size=config["obs_encoding_size"],
    #             late_fusion=config["late_fusion"],
    #             mha_num_attention_heads=config["mha_num_attention_heads"],
    #             mha_num_attention_layers=config["mha_num_attention_layers"],
    #             mha_ff_dim_factor=config["mha_ff_dim_factor"],
    #         )   
    # load_path = "/mnt/ephemeral2/Learning-to-Drive-Anywhere-via-MBRA/train/logs/frodobot-gnm/frodobot-gnm_2024_12_27_11_53_36_ExAug_GNM_FrodoBot_07/latest.pth"
    # latest_checkpoint = torch.load(load_path, map_location=torch.device(device)) 
    # model.load_state_dict(latest_checkpoint, strict=False)
    # model = model.to(device)
    # print("Model Weights Loaded")

    # # Load Dataset
    # dataset = ViNTDataset_IL2_gps_10k(repo_id=config["repo_id"], video="video_large", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=20, sacson=config["SACSoN"])
    # train_loader = DataLoader(
    #             dataset,
    #             batch_size=config["batch_size"],
    #             shuffle=False,   # don't shuffle          
    #             num_workers=12,
    #             drop_last=True,
    #             persistent_workers=True,
    #         )
    # train_iter = iter(train_loader)

    # B = config["batch_size"]
    # rsize = 0.3*torch.ones(B, 1, 1).to(device) #robot radius : 0 -- 1.0 m
    # delay = torch.zeros(B, 1, 1).to(device)   
    # linear_vel_old = 0.5*torch.ones(B, 6).float().to(device)
    # angular_vel_old = 0.0*torch.ones(B, 6).float().to(device)
    # vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)    

    # for _ in range(3):
    #     start_time  = time.time()
    #     data = next(train_iter)

    #     (
    #             image_flattened,
    #             image_goal,
    #             image_goal2,            
    #             image_current,            
    #             action_IL,
    #             goal_dist,
    #             goal_pos_relative,
    #             relative_mat,  
    #             relative_head,                        
    #             which_dataset,
    #             future_positions_unfiltered,
    #             idx,
    #             action_mask,           
    #             ped_local_slice,         
    #             ped_local_slice_raw,        
    #             ped_list_no_trans,
    #             robot_local_slice
    #         ) = data

    #     image_flattened = image_flattened.to(device)
    #     image_goal = image_goal.to(device)
    #     # idx is where we wanna WRITE 
    #     lin, ang, dist = model.forward(image_flattened, image_goal, rsize, delay, vel_past) 

    #     # output[0][:, 0].detach().cpu().numpy() # only want first velocity prediction, not all 8 predicted ones
    #     breakpoint()
    #     # print(f"ran through model and got output with len {output.shape}")

    # print("Got dataset ")

    





    
    # Set up dataset

    

    # # Create a new standalone Zarr store
    # dataset_dict_path = 'new_dataset_cache.zarr'


    dataset_dict_path = "/mnt/ephemeral2/dataset/test_cache.zarr"

    # Define dataset parameters
    total_size = 200000  # Total number of items
    chunk_size = 1000  # Adjust based on available memory
    dtype = np.float32  # Change based on your data type

    # Create a Zarr group to store multiple datasets
    zgroup = zarr.open(dataset_dict_path, mode='w')

    # Create two datasets inside the group
    z_linear = zgroup.create_dataset("linear", shape=(total_size,), chunks=(chunk_size,), dtype=dtype)
    z_angular = zgroup.create_dataset("angular", shape=(total_size,), chunks=(chunk_size,), dtype=dtype)

    # Simulate writing data in chunks
    for i in range(0, total_size, chunk_size):
        chunk_end = min(i + chunk_size, total_size)

        # Generate or load data for this chunk (replace with actual data)
        linear_chunk = np.random.rand(chunk_end - i).astype(dtype)  
        angular_chunk = np.random.rand(chunk_end - i).astype(dtype)  

        # Write to the respective datasets
        z_linear[i:chunk_end] = linear_chunk  
        z_angular[i:chunk_end] = angular_chunk  

        print(f"Written {chunk_end}/{total_size} items for both linear & angular")

    print("Finished writing dataset.")



    # # Create a new dataset and write it directly to the store
    # new_data = np.random.rand(1000)  # Example data

    # # Open the Zarr store and create a dataset inside it
    # store = zarr.open(dataset_dict_path, mode='w')  # 'w' means write mode (creates a new store)
    # store.create_dataset('new_feature', data=new_data, dtype='float64', chunks=(1000,))



    # import zarr
    # import numpy as np

    # # Define dataset parameters
    # total_size = 1203088433  # Total number of items
    # chunk_size = 1000000  # Adjust chunk size based on available memory
    # dtype = np.float32  # Change based on your data type

    # # Create a Zarr array on disk (or in memory if needed)
    # zarr_path = "dataset.zarr"
    # z = zarr.open(zarr_path, mode='w', shape=(total_size,), chunks=(chunk_size,), dtype=dtype)

    # # Simulate writing data in chunks
    # for i in range(0, total_size, chunk_size):
    #     chunk_end = min(i + chunk_size, total_size)
    #     data_chunk = np.random.rand(chunk_end - i).astype(dtype)  # Replace with actual data
    #     z[i:chunk_end] = data_chunk  # Write chunk

    #     print(f"Written {chunk_end}/{total_size} items")

    # print("Finished writing dataset.")


    # forward call 

    # model.forward(
    #     self, obs_img: torch.tensor, goal_img: torch.tensor, robot_size: torch.tensor, delay: torch.tensor, vel_past: torch.tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:


    


    # for start_idx in range(0, len(dataset), chunk_size):
    # end_idx = min(start_idx + chunk_size, len(dataset))
    
    # # Process the chunk (replace with your actual processing logic)
    # chunk_data = dataset[start_idx:end_idx]
    # new_feature_chunk = chunk_data.sum(axis=1)  # Example computation: sum of each row
    
    # # Create 'new_feature' dataset if not already present, with chunking
    # if 'new_feature' not in dataset:
    #     dataset.create_dataset('new_feature', shape=(len(dataset),), dtype='float64', chunks=(chunk_size,), overwrite=True)
    
    # # Write the new feature for this chunk
    # dataset['new_feature'][start_idx:end_idx] = new_feature_chunk
   

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)
   
    main(config)
