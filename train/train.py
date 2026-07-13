import sys
sys.path.append('../diffusion_policy')
#sys.path.append('./')

import os
import wandb
import argparse
import numpy as np
import yaml
import time
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

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.exaug.exaug import ExAug_dist_delay
from vint_train.models.il.il import IL_dist, IL_gps, IL_gps_map_mask, IL_gps_map_mask3, IL_gps_map_mask3_lan, IL_gps_map_mask3_lan2
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
try:
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
except Exception:
    ConditionalUnet1D = None  # only needed for the diffusion/NoMaD paths (not lan_only_ft)

from vint_train.models.lelan.lelan import LeLaN_clip, LeLaN_clip_temp, DenseNetwork_lelan
from vint_train.models.lelan.lelan_comp import LeLaN_clip_FiLM, LeLaN_clip_FiLM_temp

from vint_train.data.vint_dataset import ViNT_Dataset, ViNT_Dataset_fix, ViNT_Dataset_gps, ViNT_ExAug_Dataset
from vint_train.data.lelan_dataset import LeLaN_Dataset, LeLaN_Dataset_multi
from vint_train.data.vint_hf_dataset import ViNTLeRobotDataset, ViNTLeRobotDataset_annotate, ViNTDataset_annotate_10k, ViNTLeRobotDataset_IL2, ViNTLeRobotDataset_IL2_gps, ViNTDataset_IL2_gps_10k, ViNTDataset_10k, ViNTDataset_IL2_10k, ViNTLeRobotDataset_IL2_gps_map, ViNTLeRobotDataset_IL2_gps_map_crop, ViNTLeRobotDataset_IL2_gps_map2_crop, ViNTLeRobotDataset_IL2_gps_map_crop_test, EpisodeSampler_IL, EpisodeSampler_annotate, EpisodeSampler_IL_10k, EpisodeSampler_annotate_10k, ViNTDataset_IL2_gps_crop_10k, ViNTLeRobotDataset_IL2_gps_map2_crop_shadow
#from vint_train.data.frodobot_ft_dataset import Frodobot_FT_Dataset, Frodobot_FT_Dataset_shadow
from vint_train.data.bdd_dataset import BDD_Dataset_multi

from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_annotate,
    train_eval_loop_nomad,
    train_eval_loop_exaug_dist_gnm_delay,
    train_eval_loop_lelan,
    train_eval_loop_lelan_col,
    train_eval_loop_il_dist_gnm,            
    train_eval_loop_il2_dist_gnm_gps,      
    train_eval_loop_il_exaug_dist_gnm_gps, 
    train_eval_loop_il_exaug_dist_gnm_gps_map, 
    train_eval_loop_il_exaug_dist_gnm_gps_map2, 
    train_eval_loop_il_exaug_dist_gnm_gps_map2_lan,          
    train_eval_loop_il_exaug_dist_gnm_gps_map2_lan_ft,
    train_eval_loop_il_exaug_dist_gnm_gps_map2_lan_bdd,
    load_model,
)
from vint_train.training.train_utils import train_lan_only_ft, evaluate_lan_only_ft
#path_save_load = "."
path_save_load = "/nfs/kun2/users/noriaki/checkpoints"

os.environ["OMP_NUM_THREADS"] = "30"  # Set number of OpenMP threads
os.environ["MKL_NUM_THREADS"] = "30"  # Set number of MKL threads
torch.set_num_threads(30)  # Limit the number of CPU threads used by PyTorch

def main_lan_only_ft(config, device, transform):
    """
    Conservative, language-only fine-tune of OmniVLA-edge (IL_gps_map_mask3_lan2) on
    custom LeLaN-format data (`frodo_lan`) whose pickles already contain `nomad_traj_norm`.
      * builds ONLY the LeLaN loaders (no frodobot/gnm/bdd, no NoMaD/ExAug teacher)
      * optionally freezes the EfficientNet vision encoders (freeze_backbone)
      * held-out eval each epoch + early stopping on test language action-loss
      * base-vs-finetuned IMAGE-GOAL divergence to check basic driving isn't broken
    """
    dc = config["datasets_lan"]["frodo_lan"]

    def make_ds(split):
        return LeLaN_Dataset_multi(
            data_split_folder=dc[split],
            dataset_name="frodo_lan",
            image_size=config["image_size"],
            waypoint_spacing=dc.get("waypoint_spacing", 1),
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            context_size=config["context_size"],
            data_split_type=split,
            data_image_folder=dc["image"],
            data_pickle_folder=dc["pickle"],
            lan_solo=True,
            context_type=config.get("context_type", "temporal"),
            normalize=config["normalize"],
            backside=dc.get("backside", False),
            aug_seq=dc.get("aug_seq", False),
            only_front=dc.get("only_front", True),
        )

    train_loader = DataLoader(
        make_ds("train"), batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], drop_last=True,
    )
    test_loader = DataLoader(
        make_ds("test"), batch_size=config.get("eval_batch_size", config["batch_size"]),
        shuffle=False, num_workers=config["num_workers"], drop_last=False,
    )

    def build_model():
        return IL_gps_map_mask3_lan2(
            context_size=config["context_size"], len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"], obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"], late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )

    def load_ckpt(m, path):
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        return m.load_state_dict(sd, strict=False)

    ckpt_path = config["load_edge_ckpt"]
    print("Loading edge checkpoint from", ckpt_path)
    model = build_model()
    missing, unexpected = load_ckpt(model, ckpt_path)
    print(f"[ckpt] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 10):", missing[:10])
    if unexpected:
        print("  unexpected (first 10):", unexpected[:10])
    model = model.to(device)

    # frozen reference copy of the ORIGINAL weights for the basic-driving regression check
    base_model = build_model()
    load_ckpt(base_model, ckpt_path)
    base_model = base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # ---- freeze the vision backbones (recommended for tiny fine-tune sets) ----
    freeze_backbone = config.get("freeze_backbone", True)
    if freeze_backbone:
        for enc in (model.obs_encoder, model.goal_encoder, model.goal_encoder_img):
            for p in enc.parameters():
                p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M "
          f"(freeze_backbone={freeze_backbone})")

    text_encoder, _ = clip.load(config["clip_type"])
    text_encoder.to(torch.float32).to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    if config.get("warmup", True):
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=config.get("warmup_epochs", 2), after_scheduler=scheduler
        )

    out = config.get("output_dir", "./logs_frodo_lan_ft")
    os.makedirs(out, exist_ok=True)
    print("Saving fine-tuned checkpoints to", out)

    patience = int(config.get("early_stop_patience", 3))
    best_loss = float("inf")
    best_epoch = -1
    since_improve = 0

    for epoch in range(config["epochs"]):
        train_lan_only_ft(
            model=model, text_encoder=text_encoder, optimizer=optimizer,
            dataloader_lan=train_loader, transform=transform, device=device,
            project_folder=out, epoch=epoch,
            print_log_freq=config.get("print_log_freq", 20),
            use_wandb=config["use_wandb"], freeze_backbone=freeze_backbone,
        )
        if scheduler is not None:
            scheduler.step()

        metrics = evaluate_lan_only_ft(model, base_model, text_encoder, test_loader, transform, device)
        print(f"[eval] epoch {epoch}  test_action_loss={metrics['test_action_loss']:.4f}  "
              f"test_obj_loss={metrics['test_obj_loss']:.4f}  "
              f"base_divergence(image-goal)={metrics['base_divergence_imagegoal']:.4f}")
        if config["use_wandb"]:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

        torch.save(model.state_dict(), os.path.join(out, "latest.pth"))

        cur = metrics["test_action_loss"]
        if cur < best_loss:
            best_loss = cur
            best_epoch = epoch
            since_improve = 0
            torch.save(model.state_dict(), os.path.join(out, "best.pth"))
            print(f"[save] new best (test_action_loss={best_loss:.4f}) -> {out}/best.pth")
        else:
            since_improve += 1
            print(f"[early-stop] no improvement {since_improve}/{patience} "
                  f"(best={best_loss:.4f} @ epoch {best_epoch})")
            if since_improve >= patience:
                print(f"[early-stop] stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

    print(f"FINISHED LAN-ONLY FINE-TUNE (best test_action_loss={best_loss:.4f} "
          f"@ epoch {best_epoch}) -> {out}/best.pth")


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # ---- language-only fine-tune path (frodo_lan): no frodobot/gnm/bdd data, no NoMaD ----
    if config.get("lan_only_ft", False):
        main_lan_only_ft(config, device, transform)
        return

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    if "raw_label" not in config:
        config["raw_label"] = False
    print("raw_label", config["raw_label"])

    if "lan_solo" not in config:
        config["lan_solo"] = False
    if "sate_solo" not in config:
        config["sate_solo"] = False        
    if "image_solo" not in config:
        config["image_solo"] = False
    if "no_sate" not in config:
        config["no_sate"] = False 
    if "ft_frod" not in config:
        config["ft_frod"] = False                        
    if "aug_shadow" not in config:
        config["aug_shadow"] = False  
    if "bdd" not in config:
        config["bdd"] = False                
    if config["image_solo"] == True:
        config["horizon_long"] = 20
                
    for dataset_name in config["datasets"]:
        print("dataset_name", dataset_name)
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        train_dataset_lan = []
        test_dataset_lan = []
        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                if config["project_name"] == "lelan-release":                  
                    dataset = LeLaN_Dataset(
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size_clip"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        data_split_type = data_split_type,
                        data_image_folder = data_config["image"],
                        data_pickle_folder = data_config["pickle"],                                         
                        context_type=config["context_type"],
                        normalize=config["normalize"],
                        backside=data_config["backside"],
                        aug_seq=data_config["aug_seq"],   
                        only_front=data_config["only_front"],                                                      
                    )  
                elif config["project_name"] == "frodobot-gnm-lelan":  
                    if config["ft_frod"]:
                        dataset_name_ft = "frodobot_ft"
                        data_config_ft = config["datasets_ft"][dataset_name_ft]   
                        if config["aug_shadow"]:
                            dataset_ft = Frodobot_FT_Dataset_shadow(
                                data_split_folder=data_config_ft[data_split_type],
                                dataset_name=dataset_name_ft,
                                image_size=config["image_size"],
                                waypoint_spacing=data_config_ft["waypoint_spacing"],
                                len_traj_pred=config["len_traj_pred"],
                                learn_angle=config["learn_angle"],
                                context_size=config["context_size"],
                                data_split_type = data_split_type,
                                data_image_folder = data_config_ft["image"],
                                data_pickle_folder = data_config_ft["pickle"],                                             
                                context_type=config["context_type"],
                                normalize=config["normalize"],
                                aug_seq=data_config_ft["aug_seq"],   
                                only_front=data_config_ft["only_front"],                                                      
                            )
                        else:
                            dataset_ft = Frodobot_FT_Dataset(
                                data_split_folder=data_config_ft[data_split_type],
                                dataset_name=dataset_name_ft,
                                image_size=config["image_size"],
                                waypoint_spacing=data_config_ft["waypoint_spacing"],
                                len_traj_pred=config["len_traj_pred"],
                                learn_angle=config["learn_angle"],
                                context_size=config["context_size"],
                                data_split_type = data_split_type,
                                data_image_folder = data_config_ft["image"],
                                data_pickle_folder = data_config_ft["pickle"],                                             
                                context_type=config["context_type"],
                                normalize=config["normalize"],
                                aug_seq=data_config_ft["aug_seq"],   
                                only_front=data_config_ft["only_front"],                                                      
                            )                                                         
                        batch_ft = 40
                    else:
                        batch_ft = 0                      

                    ##test BDD dataloader##
                    if config["bdd"]:
                        data_config_bdd = config["datasets_bdd"]
                        dataset_bdd = BDD_Dataset_multi(
                            data_split_folder=data_config_bdd[data_split_type],
                            dataset_name="bdd",
                            image_size=config["image_size"],
                            waypoint_spacing=data_config_bdd["waypoint_spacing"],
                            len_traj_pred=config["len_traj_pred"],
                            learn_angle=config["learn_angle"],
                            context_size=config["context_size"],
                            data_split_type = data_split_type,
                            data_folder = data_config_bdd["image"],                                                               
                            context_type=config["context_type"],
                            normalize=config["normalize"],
                            model_type=config["model_type"],
                            aug_seq=data_config_bdd["aug_seq"],                                                   
                        )
                    else:
                        batch_bdd = 0
                
                    for dataset_name_lan in config["datasets_lan"]:
                        data_config_lan = config["datasets_lan"][dataset_name_lan]   
                        if "negative_mining" not in data_config_lan:
                            data_config_lan["negative_mining"] = True
                        if "goals_per_obs" not in data_config_lan:
                            data_config_lan["goals_per_obs"] = 1
                        if "end_slack" not in data_config_lan:
                            data_config_lan["end_slack"] = 0
                        if "waypoint_spacing" not in data_config_lan:
                            data_config_lan["waypoint_spacing"] = 1
                                                 
                        dataset_lan = LeLaN_Dataset_multi(
                            data_split_folder=data_config_lan[data_split_type],
                            dataset_name=dataset_name_lan,
                            image_size=config["image_size"],
                            waypoint_spacing=data_config_lan["waypoint_spacing"],
                            len_traj_pred=config["len_traj_pred"],
                            learn_angle=config["learn_angle"],
                            context_size=config["context_size"],
                            data_split_type = data_split_type,
                            data_image_folder = data_config_lan["image"],
                            data_pickle_folder = data_config_lan["pickle"], 
                            lan_solo = config["lan_solo"],                                                                         
                            context_type=config["context_type"],
                            normalize=config["normalize"],
                            backside=data_config_lan["backside"],
                            aug_seq=data_config_lan["aug_seq"],   
                            only_front=data_config_lan["only_front"],                                                      
                        ) 
                        if data_split_type == "train":
                            train_dataset_lan.append(dataset_lan)
                        elif data_split_type == "test":
                            test_dataset_lan.append(dataset_lan)
                                                    
                    if config["lan_solo"]:
                        batch_frodobot = 2 #dummy
                        batch_gnm = 2 #dummy
                        batch_lelan = config["batch_size"]    
                    elif config["sate_solo"]:
                        batch_frodobot = config["batch_size"] 
                        batch_gnm = 2 #dummy
                        batch_lelan = 2 #dummy      
                    elif config["bdd"]:
                        batch_bdd = int((config["batch_size"])/4.0) 
                        batch_frodobot = int((config["batch_size"])/4.0) 
                        batch_gnm = int((config["batch_size"])/4.0)
                        batch_lelan = int(config["batch_size"] - batch_bdd - batch_frodobot - batch_gnm)                                                    
                    else:
                        batch_frodobot = int((config["batch_size"]-batch_ft)/3.0) 
                        batch_gnm = int((config["batch_size"]-batch_ft)/3.0)
                        batch_lelan = int(config["batch_size"] - batch_ft - batch_frodobot - batch_gnm)                           
                                                
                    #split_train_test = 100 #[TODO] we do not need to setup split_train_test.
                    ratio_f = config["ratio_f"]
                    split_train_test = int(11994*ratio_f)                     
                    if data_split_type == "train": 
                        if not config["10k_data"]:    
                            if config["aug_shadow"]:
                                dataset = ViNTLeRobotDataset_IL2_gps_map2_crop_shadow(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                               
                            else:
                                dataset = ViNTLeRobotDataset_IL2_gps_map2_crop(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                                             
                            episode_sampler_train = EpisodeSampler_IL(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)   
                    else:
                        if not config["10k_data"]: 
                            if config["aug_shadow"]:
                                dataset = ViNTLeRobotDataset_IL2_gps_map2_crop_shadow(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                             
                            else:                                             
                                dataset = ViNTLeRobotDataset_IL2_gps_map2_crop(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                
                            episode_sampler_test = EpisodeSampler_IL(dataset, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type)              
                                                                                                
                elif config["project_name"] == "frodobot":
                    if data_split_type == "train":                
                        """
                        delta_f = int((187156-1)/30)
                        ic = 0
                        start_ep = ic*delta_f
                        end_ep = (ic+1)*delta_f
                        if start_ep > 187156-1:
                            start_ep = 187156-1                        
                        if end_ep > 187156:
                            end_ep = 187156
                        """
                        start_ep = 0
                        end_ep = 187156
                        print("start-end episode", start_ep, end_ep)
                        if config["model_type"] == "annotate":
                            if config["10k_data"]:                        
                                dataset = ViNTDataset_annotate_10k(repo_id=config["repo_id"], video="video_large", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=20, sacson=config["SACSoN"])
                                episode_sampler_train = EpisodeSampler_annotate(dataset, start_ep, end_ep-1, goal_horizon=20, data_split_type=data_split_type) 
                            else:                        
                                dataset = ViNTLeRobotDataset_annotate(repo_id=config["repo_id"], video="video_large", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=20, sacson=config["SACSoN"])
                                episode_sampler_train = EpisodeSampler_annotate(dataset, 0, 11994-1, goal_horizon=20, data_split_type=data_split_type)                  
                    elif data_split_type == "test":
                        if config["model_type"] == "annotate":
                            if config["10k_data"]:                             
                                dataset = ViNTDataset_annotate_10k(repo_id=config["repo_id"], video="video_large", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=20, sacson=config["SACSoN"])                                                   
                                episode_sampler_test = EpisodeSampler_annotate(dataset, start_ep, end_ep-1, goal_horizon=20, data_split_type=data_split_type)               
                            else:
                                dataset = ViNTLeRobotDataset_annotate(repo_id=config["repo_id"], video="video_large", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=20, sacson=config["SACSoN"])                                                   
                                episode_sampler_test = EpisodeSampler_annotate(dataset, 11994-1, 11994-1, goal_horizon=20, data_split_type=data_split_type)                                                            
                elif config["project_name"] == "frodobot-gnm":  
                    if config["10k_data"]:
                        ratio_f = config["ratio_f"]
                        split_train_test = int(187156*ratio_f)
                        batch_frodobot = 150
                        batch_gnm = int(config["batch_size"] - batch_frodobot)                                              
                    else:                  
                        ratio_f = config["ratio_f"]
                        split_train_test = int(11994*ratio_f)
                        batch_gnm = int(config["batch_size"] * (835840/4)/(835840/4 + ratio_f*11485100.0/10.0))
                        batch_frodobot = int(config["batch_size"] - batch_gnm)    
                        batch_frodobot = int(config["batch_size"]*0.5) 
                        batch_gnm = int(config["batch_size"] - batch_frodobot)                   
                    
                    print("batch_gnm", batch_gnm, "batch_frodobot", batch_frodobot)                                            
                    print("frodobot", data_split_type, data_config[data_split_type])                  
                    
                    if data_split_type == "train":                   
                        if config["model_type"] == "il_dist_gnm":
                            if config["10k_data"]:
                                dataset = ViNTDataset_IL2_10k(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                                          
                                episode_sampler_train = EpisodeSampler_IL_10k(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                            
                            else:                        
                                dataset = ViNTLeRobotDataset_IL2(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                                          
                                episode_sampler_train = EpisodeSampler_IL(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)       
                        elif config["model_type"] == "il2_gps" or config["model_type"] == "il_exaug_gps":      
                            if config["10k_data"]:
                                #dataset = ViNTDataset_IL2_gps_crop(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)  
                                dataset = ViNTDataset_IL2_gps_crop_10k(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                                          
                                episode_sampler_train = EpisodeSampler_IL_10k(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                                         
                            else:
                                dataset = ViNTLeRobotDataset_IL2_gps(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                                          
                                episode_sampler_train = EpisodeSampler_IL(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)             
                        elif config["model_type"] == "il_exaug_gps_map" or config["model_type"] == "il_exaug_gps_map2":        
                            dataset = ViNTLeRobotDataset_IL2_gps_map2_crop(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                                          
                            episode_sampler_train = EpisodeSampler_IL(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                                     
                        else:
                            if config["10k_data"]:
                                dataset = ViNTDataset_10k(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=1)                                                                                        
                                episode_sampler_train = EpisodeSampler_IL_10k(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                                                          
                            else:
                                dataset = ViNTLeRobotDataset(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=1)                                                                                        
                                episode_sampler_train = EpisodeSampler_IL(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                                                 
                    elif data_split_type == "test":              
                        if config["model_type"] == "il_dist_gnm":
                            if config["10k_data"]:
                                dataset = ViNTDataset_IL2_10k(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                  
                                episode_sampler_test = EpisodeSampler_IL_10k(dataset, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type)  
                            else:
                                dataset = ViNTLeRobotDataset_IL2(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                  
                                episode_sampler_test = EpisodeSampler_IL(dataset, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                              
                        elif config["model_type"] == "il2_gps" or config["model_type"] == "il_exaug_gps":
                            if config["10k_data"]:
                                dataset = ViNTDataset_IL2_gps_crop_10k(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                                           
                                episode_sampler_test = EpisodeSampler_IL_10k(dataset, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                               
                            else:
                                dataset = ViNTLeRobotDataset_IL2_gps(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                  
                                episode_sampler_test = EpisodeSampler_IL(dataset, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type)          
                        elif config["model_type"] == "il_exaug_gps_map" or config["model_type"] == "il_exaug_gps_map2":  
                            dataset = ViNTLeRobotDataset_IL2_gps_map2_crop(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], goal_horizon2=config["horizon_long"], sacson=config["SACSoN"], context_spacing=3, action_spacing=3)                
                            episode_sampler_test = EpisodeSampler_IL(dataset, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                                                                           
                        else:
                            if config["10k_data"]:
                                dataset = ViNTDataset_10k(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=1)           #                                                                             
                                episode_sampler_test = EpisodeSampler_IL_10k(dataset, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                               
                            if not config["10k_data"]:
                                dataset = ViNTLeRobotDataset(repo_id=config["repo_id"], video="video", root=config["root"], image_size=config["image_size"], split="train", goal_horizon=config["horizon_short"], sacson=config["SACSoN"], context_spacing=3, action_spacing=1)                                                 
                                episode_sampler_test = EpisodeSampler_IL(dataset, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type)                                               
                else:                
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                    if config["model_type"] == "il_exaug_gps_map2_lan":
                        #train_dataset_lan = dataset_lan                    
                        train_dataset_lan = ConcatDataset(train_dataset_lan)  
                    if config["ft_frod"]:
                        train_dataset_ft = dataset_ft                                          
                    if config["bdd"]:
                        train_dataset_bdd = dataset_bdd                            
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset
                    if config["model_type"] == "il_exaug_gps_map2_lan":                    
                        #test_dataset_lan = dataset_lan
                        test_dataset_lan = ConcatDataset(test_dataset_lan)          
                    if config["ft_frod"]:
                        test_dataset_ft = dataset_ft
                    if config["bdd"]:
                        test_dataset_bdd = dataset_bdd                          
                                            
    train_dataset_sub = []
    test_dataloaders_sub = {}    
    if config["project_name"] == "frodobot-gnm" or config["project_name"] == "frodobot-gnm-lelan":             
        for dataset_name in config["datasets_sub"]:                
            data_config_sub = config["datasets_sub"][dataset_name]
            if "negative_mining" not in data_config_sub:
                data_config_sub["negative_mining"] = True
            if "goals_per_obs" not in data_config_sub:
                data_config_sub["goals_per_obs"] = 1
            if "end_slack" not in data_config_sub:
                data_config_sub["end_slack"] = 0
            if "waypoint_spacing" not in data_config_sub:
                data_config_sub["waypoint_spacing"] = 1            

            for data_split_type in ["train", "test"]:
                if data_split_type in data_config_sub:
                    if config["model_type"] == "exaug_dist_gnm_delay":            
                        dataset = ViNT_ExAug_Dataset(
                            data_folder=data_config_sub["data_folder"],
                            data_split_folder=data_config_sub[data_split_type],
                            dataset_name=dataset_name,
                            image_size=config["image_size"],
                            waypoint_spacing=data_config_sub["waypoint_spacing"],
                            min_dist_cat=config["distance"]["min_dist_cat"],
                            max_dist_cat=config["distance"]["max_dist_cat"],
                            min_action_distance=config["action"]["min_dist_cat"],
                            max_action_distance=config["action"]["max_dist_cat"],
                            negative_mining=data_config_sub["negative_mining"],
                            len_traj_pred=config["len_traj_pred"],
                            learn_angle=config["learn_angle"],
                            context_size=config["context_size"],
                            context_type=config["context_type"],
                            end_slack=data_config_sub["end_slack"],
                            goals_per_obs=data_config_sub["goals_per_obs"],
                            normalize=config["normalize"],
                            goal_type=config["goal_type"],
                        )
                    elif config["model_type"] == "il2_gps" or config["model_type"] == "il_exaug_gps" or config["model_type"] == "il_exaug_gps_map" or config["model_type"] == "il_exaug_gps_map2" or config["model_type"] == "il_exaug_gps_map2_lan":
                        dataset = ViNT_Dataset_gps(
                            data_folder=data_config_sub["data_folder"],
                            data_split_folder=data_config_sub[data_split_type],
                            dataset_name=dataset_name,
                            image_size=config["image_size"],
                            waypoint_spacing=data_config_sub["waypoint_spacing"],
                            min_dist_cat=config["distance"]["min_dist_cat"],
                            max_dist_cat=config["distance"]["max_dist_cat"],
                            min_action_distance=config["action"]["min_dist_cat"],
                            max_action_distance=config["action"]["max_dist_cat"],
                            negative_mining=data_config_sub["negative_mining"],
                            len_traj_pred=config["len_traj_pred"],
                            learn_angle=config["learn_angle"],
                            context_size=config["context_size"],
                            context_type=config["context_type"],
                            end_slack=data_config_sub["end_slack"],
                            goals_per_obs=data_config_sub["goals_per_obs"],
                            normalize=config["normalize"],
                            goal_type=config["goal_type"],
                        )                                                    
                    else:
                        dataset = ViNT_Dataset(
                            data_folder=data_config_sub["data_folder"],
                            data_split_folder=data_config_sub[data_split_type],
                            dataset_name=dataset_name,
                            image_size=config["image_size"],
                            waypoint_spacing=data_config_sub["waypoint_spacing"],
                            min_dist_cat=config["distance"]["min_dist_cat"],
                            max_dist_cat=config["distance"]["max_dist_cat"],
                            min_action_distance=config["action"]["min_dist_cat"],
                            max_action_distance=config["action"]["max_dist_cat"],
                            negative_mining=data_config_sub["negative_mining"],
                            len_traj_pred=config["len_traj_pred"],
                            learn_angle=config["learn_angle"],
                            context_size=config["context_size"],
                            context_type=config["context_type"],
                            end_slack=data_config_sub["end_slack"],
                            goals_per_obs=data_config_sub["goals_per_obs"],
                            normalize=config["normalize"],
                            goal_type=config["goal_type"],
                        )                        
                    if data_split_type == "train":
                        train_dataset_sub.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders_sub:
                            test_dataloaders_sub[dataset_type] = {}
                        test_dataloaders_sub[dataset_type] = dataset
        train_dataset_sub = ConcatDataset(train_dataset_sub)

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)
    #print(len(episode_sampler_train), len(episode_sampler_test))

    if config["project_name"] == "frodobot-gnm":
        if config["10k_data"]:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_frodobot,
                shuffle=False,            
                num_workers=config["num_workers_f"],
                drop_last=True,
                persistent_workers=True,
                sampler=episode_sampler_train,
            )        
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_frodobot,
                shuffle=False,            
                num_workers=config["num_workers_f"],
                drop_last=True,
                persistent_workers=True,
                sampler=episode_sampler_train,
            )
        train_loader_sub = DataLoader(
            train_dataset_sub,
            batch_size=batch_gnm,
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=True,
            persistent_workers=True,
        )        

    elif config["project_name"] == "frodobot-gnm-lelan":
        if config["10k_data"]:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_frodobot,
                shuffle=False,            
                num_workers=config["num_workers"],
                drop_last=True,
                persistent_workers=True,
                sampler=episode_sampler_train,
            )        
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_frodobot,
                shuffle=False,            
                num_workers=config["num_workers"],
                drop_last=True,
                persistent_workers=True,
                sampler=episode_sampler_train,
            )
        train_loader_sub = DataLoader(
            train_dataset_sub,
            batch_size=batch_gnm,
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=True,
            persistent_workers=True,
        )                 
        train_loader_lelan = DataLoader(
            train_dataset_lan,
            batch_size=batch_lelan,
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=True,
            persistent_workers=True,
        )    
        if config["ft_frod"]:
            train_loader_ft = DataLoader(
                train_dataset_ft,
                batch_size=batch_ft,
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
                persistent_workers=True,
            )                 
        if config["bdd"]:
            train_loader_bdd = DataLoader(
                train_dataset_bdd,
                batch_size=batch_bdd,
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
                persistent_workers=True,
            )                                  
    elif config["project_name"] == "frodobot":
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
        )        
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
        )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    if config["project_name"] == "frodobot-gnm":
        if config["10k_data"]:    
            for dataset_type, dataset in test_dataloaders.items():
                test_dataloaders[dataset_type] = DataLoader(
                    dataset,
                    batch_size=config["eval_batch_size"],
                    shuffle=False,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    sampler=episode_sampler_test,              
                )    
        else: 
            for dataset_type, dataset in test_dataloaders.items():
                test_dataloaders[dataset_type] = DataLoader(
                    dataset,
                    batch_size=config["eval_batch_size"],
                    shuffle=False,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    sampler=episode_sampler_test,                
                )
        for dataset_type, dataset in test_dataloaders_sub.items():
            test_dataloaders_sub[dataset_type] = DataLoader(
                dataset,
                batch_size=config["eval_batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
            )    

    elif config["project_name"] == "frodobot-gnm-lelan":
        if config["10k_data"]:    
            for dataset_type, dataset in test_dataloaders.items():
                test_dataloaders[dataset_type] = DataLoader(
                    dataset,
                    batch_size=config["eval_batch_size"],
                    shuffle=False,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    sampler=episode_sampler_test,              
                )    
        else: 
            for dataset_type, dataset in test_dataloaders.items():
                test_dataloaders[dataset_type] = DataLoader(
                    dataset,
                    batch_size=config["eval_batch_size"],
                    shuffle=False,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    sampler=episode_sampler_test,                
                )
        for dataset_type, dataset in test_dataloaders_sub.items():
            test_dataloaders_sub[dataset_type] = DataLoader(
                dataset,
                batch_size=config["eval_batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
            )   
        test_loader_lelan = DataLoader(
            test_dataset_lan,
            batch_size=batch_lelan,
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=True,
            persistent_workers=True,
        )                     
        if config["ft_frod"]:
            test_loader_ft = DataLoader(
                test_dataset_ft,
                batch_size=batch_ft,
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
                persistent_workers=True,
            )               
        if config["bdd"]:
            test_loader_bdd = DataLoader(
                test_dataset_bdd,
                batch_size=batch_bdd,
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
                persistent_workers=True,
            )                                      
    else:
        for dataset_type, dataset in test_dataloaders.items():
            test_dataloaders[dataset_type] = DataLoader(
                dataset,
                batch_size=config["eval_batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=False,
            )    

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
     
    elif config["model_type"] == "exaug_dist_gnm_delay":
        model = ExAug_dist_delay(       
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )         
    elif config["model_type"] == "il_dist" or config["model_type"] == "il_dist_gnm" or config["model_type"] == "il_dist_gnm_vis":
        model = IL_dist(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        ) 
    elif config["model_type"] == "il2_dist_gnm":
        model_GNM = IL_dist(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        ) 
        model = IL_dist(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )         

    elif config["model_type"] == "il_exaug_dist_gnm":
        model_GNM = ExAug_dist_delay(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )               
        model = IL_dist(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )  

    elif config["model_type"] == "il2_gps":
        model_GNM = IL_dist(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        ) 
        model = IL_gps(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )   

    elif config["model_type"] == "il_exaug_gps":
        model_GNM = ExAug_dist_delay(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )    
        model = IL_gps(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )   

    elif config["model_type"] == "il_exaug_gps_map":
        model_GNM = ExAug_dist_delay(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )    
        model = IL_gps_map_mask(        
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )  
    elif config["model_type"] == "il_exaug_gps_map2":
        model_GNM = ExAug_dist_delay(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )    
        #model = IL_gps_map(
        model = IL_gps_map_mask3(        
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )    
    elif config["model_type"] == "il_exaug_gps_map2_lan":
        if config["bdd"]:
            model_bdd = ExAug_dist_delay(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                late_fusion=config["late_fusion"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )               
        model_GNM = ExAug_dist_delay(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )    
        #model = IL_gps_map(
        model = IL_gps_map_mask3_lan2(        
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )  
        #print(torch.cuda.device_count())
        text_encoder, preprocess = clip.load(config["clip_type"])    
        text_encoder.to(torch.float32)             

        vision_encoder_nomad = NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"]-4,
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        vision_encoder_nomad = replace_bn_with_gn(vision_encoder_nomad)

        noise_pred_net_nomad = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network_nomad = DenseNetwork(embedding_dim=config["encoding_size"])

        model_nomad = NoMaD(
            vision_encoder=vision_encoder_nomad,
            noise_pred_net=noise_pred_net_nomad,
            dist_pred_net=dist_pred_network_nomad,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    elif config["model_type"] == "annotate":
        model = []                                          
    elif config["model_type"] == "lelan":
        if config["vision_encoder"] == "lelan_clip_film":
            vision_encoder = LeLaN_clip_FiLM(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                feature_size=config["feature_size"],
                clip_type=config["clip_type"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)   
            
            text_encoder, preprocess = clip.load(config["clip_type"])  

        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        dist_pred_network = DenseNetwork_lelan(embedding_dim=config["encoding_size"], control_horizon=config["len_traj_pred"])
                  
        if config["vision_encoder"] == "lelan_clip_film":
            model = LeLaN_clip(
                vision_encoder=vision_encoder,
                dist_pred_net=dist_pred_network,
                text_encoder=text_encoder
            )                
          
    elif config["model_type"] == "lelan_col":
        if config["vision_encoder"] == "lelan_clip_film":
            vision_encoder = LeLaN_clip_FiLM_temp(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                feature_size=config["feature_size"],
                clip_type=config["clip_type"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)    
            text_encoder, preprocess = clip.load(config["clip_type"])    
            text_encoder.to(torch.float32)
            
            vision_encoder_nomad = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder_nomad = replace_bn_with_gn(vision_encoder_nomad)

            noise_pred_net_nomad = ConditionalUnet1D(
                    input_dim=2,
                    global_cond_dim=config["encoding_size"],
                    down_dims=config["down_dims"],
                    cond_predict_scale=config["cond_predict_scale"],
                )
            dist_pred_network_nomad = DenseNetwork(embedding_dim=config["encoding_size"])
            dist_pred_network = DenseNetwork_lelan(embedding_dim=config["encoding_size"], control_horizon=config["len_traj_pred"])
            
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        model = LeLaN_clip_temp(
            vision_encoder=vision_encoder,
            dist_pred_net=dist_pred_network,            
            text_encoder=text_encoder,
        )
        
        model_nomad = NoMaD(
            vision_encoder=vision_encoder_nomad,
            noise_pred_net=noise_pred_net_nomad,
            dist_pred_net=dist_pred_network_nomad,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
                        
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    
    if config["model_type"] != "annotate":
        if config["optimizer"] == "adam":
            optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
        elif config["optimizer"] == "adamw":
            optimizer = AdamW(model.parameters(), lr=lr)
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {config['optimizer']} not supported")

        scheduler = None
        if config["scheduler"] is not None:
            config["scheduler"] = config["scheduler"].lower()
            if config["scheduler"] == "cosine":
                print("Using cosine annealing with T_max", config["epochs"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config["epochs"]
                )
            elif config["scheduler"] == "cyclic":
                print("Using cyclic LR with cycle", config["cyclic_period"])
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=lr / 10.,
                    max_lr=lr,
                    step_size_up=config["cyclic_period"] // 2,
                    cycle_momentum=False,
                )
            elif config["scheduler"] == "plateau":
                print("Using ReduceLROnPlateau")
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=config["plateau_factor"],
                    patience=config["plateau_patience"],
                    verbose=True,
                )
            else:
                raise ValueError(f"Scheduler {config['scheduler']} not supported")

            if config["warmup"]:
                print("Using warmup scheduler")
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=config["warmup_epochs"],
                    after_scheduler=scheduler,
                )
                
        def count_files(directory):
            return sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))
                

        current_epoch = 0
        if "load_run" in config:
            load_project_folder = os.path.join(path_save_load, "logs", config["load_run"])
            print("Loading model from ", load_project_folder)
            latest_path = os.path.join(load_project_folder, "latest.pth")
            latest_checkpoint = torch.load(latest_path, map_location=torch.device(device)) 
            load_model(model, config["model_type"], latest_checkpoint)
            file_count = count_files(load_project_folder)
            current_epoch = int((file_count - 3)) 

        if "load_exaug" in config:
            load_project_folder_exaug = os.path.join(config["load_exaug"])
            print("Loading ExAug model from ", load_project_folder_exaug)
            if "model_10k" in config:
                if config["model_10k"]:
                    latest_path_exaug = os.path.join(load_project_folder_exaug, "exaug_labeler_10k.pth")
            else:
                latest_path_exaug = os.path.join(load_project_folder_exaug, "exaug_labeler.pth")
            latest_checkpoint_exaug = torch.load(latest_path_exaug, map_location=torch.device(device)) 
            load_model(model_GNM, config["model_type"], latest_checkpoint_exaug)
            model_GNM.eval().to(device)

        if "load_bdd" in config:
            load_project_folder_bdd = os.path.join(config["load_bdd"])
            print("Loading ExAug BDD model from ", load_project_folder_bdd)
            latest_path_bdd = os.path.join(load_project_folder_bdd, "exaug_bdd_labeler.pth")
            latest_checkpoint_bdd = torch.load(latest_path_bdd, map_location=torch.device(device)) 
            load_model(model_bdd, config["model_type"], latest_checkpoint_bdd)
            model_bdd.eval().to(device)

        if "load_il" in config:
            load_project_folder_IL = os.path.join(config["load_il"])
            print("Loading IL model from ", load_project_folder_IL)
            latest_path_IL = os.path.join(load_project_folder_IL, "il_labeler.pth")
            latest_checkpoint_IL = torch.load(latest_path_IL, map_location=torch.device(device)) 
            load_model(model_GNM, config["model_type"], latest_checkpoint_IL)
            model_GNM.eval().to(device)

        if "load_nomad" in config:
            load_project_folder = os.path.join("logs", config["load_nomad"])
            print("Loading NoMaD model from ", load_project_folder)
            latest_path = os.path.join(load_project_folder, "nomad_crop.pth")
            latest_checkpoint = torch.load(latest_path, map_location=torch.device(device)) 
            load_model(model_nomad, config["model_type"], latest_checkpoint)
            model_nomad.to(device)

        # Multi-GPU
        if len(config["gpu_ids"]) > 1:
            model = nn.DataParallel(model, device_ids=config["gpu_ids"])
        model = model.to(device)
        if "load_run" in config:  # load optimizer and scheduler after data parallel
            if "optimizer" in latest_checkpoint:
                optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
            if scheduler is not None and "scheduler" in latest_checkpoint:
                scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    else:
        current_epoch = 0
        optimizer = []
        scheduler = []

    print("current_epoch", current_epoch)
    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    elif config["model_type"] == "lelan":
        train_eval_loop_lelan(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            save_freq=config["save_freq"],
        )    
    elif config["model_type"] == "lelan_col":
        train_eval_loop_lelan_col(
            train_model=config["train"],
            model=model,
            model_nomad=model_nomad,            
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            weight_col_loss=config["weight_col_loss"],            
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            save_freq=config["save_freq"],
        )            
    elif config["model_type"] == "exaug_dist_gnm_delay":
        train_eval_loop_exaug_dist_gnm_delay(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            train_loader_sub=train_loader_sub,
            test_dataloaders_sub=test_dataloaders_sub,                  
            transform=transform,
            epochs=config["epochs"],
            sacson=config["SACSoN"],
            device=device,
            batch_size=config["batch_size"],
            batch_size_test=config["eval_batch_size"], 
            len_traj_pred=config["len_traj_pred"],           
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )          
    elif config["model_type"] == "il_dist_gnm":        
        train_eval_loop_il_dist_gnm(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            train_loader_sub=train_loader_sub,
            test_dataloaders_sub=test_dataloaders_sub,            
            transform=transform,
            epochs=config["epochs"],
            sacson=config["SACSoN"],
            device=device,
            batch_size=config["batch_size"],
            batch_size_test=config["eval_batch_size"], 
            len_traj_pred=config["len_traj_pred"],           
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )            
    elif config["model_type"] == "il2_gps":
        train_eval_loop_il2_dist_gnm_gps(
            train_model=config["train"],
            model=model,
            model_GNM=model_GNM,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            train_loader_sub=train_loader_sub,
            test_dataloaders_sub=test_dataloaders_sub,            
            transform=transform,
            epochs=config["epochs"],
            sacson=config["SACSoN"],
            device=device,
            batch_size=config["batch_size"],
            batch_size_test=config["eval_batch_size"], 
            len_traj_pred=config["len_traj_pred"],           
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )                  
    elif config["model_type"] == "il_exaug_gps":
        train_eval_loop_il_exaug_dist_gnm_gps(
            train_model=config["train"],
            model=model,
            model_GNM=model_GNM,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            train_loader_sub=train_loader_sub,
            test_dataloaders_sub=test_dataloaders_sub,            
            transform=transform,
            epochs=config["epochs"],
            sacson=config["SACSoN"],
            device=device,
            batch_size=config["batch_size"],
            batch_size_test=config["eval_batch_size"], 
            len_traj_pred=config["len_traj_pred"],           
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )   
    elif config["model_type"] == "il_exaug_gps_map":      
        train_eval_loop_il_exaug_dist_gnm_gps_map(
            train_model=config["train"],
            model=model,
            model_GNM=model_GNM,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            train_loader_sub=train_loader_sub,
            test_dataloaders_sub=test_dataloaders_sub,            
            transform=transform,
            epochs=config["epochs"],
            sacson=config["SACSoN"],
            device=device,
            batch_size=config["batch_size"],
            batch_size_test=config["eval_batch_size"], 
            len_traj_pred=config["len_traj_pred"],           
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )   
    elif config["model_type"] == "il_exaug_gps_map2":
        train_eval_loop_il_exaug_dist_gnm_gps_map2(
            train_model=config["train"],
            model=model,
            model_GNM=model_GNM,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            #noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            train_loader_sub=train_loader_sub,
            test_dataloaders_sub=test_dataloaders_sub,            
            transform=transform,
            #goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            sacson=config["SACSoN"],
            device=device,
            batch_size=config["batch_size"],
            batch_size_test=config["eval_batch_size"], 
            len_traj_pred=config["len_traj_pred"],           
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )     
    elif config["model_type"] == "il_exaug_gps_map2_lan":
        if config["ft_frod"]:
            train_eval_loop_il_exaug_dist_gnm_gps_map2_lan_ft(
                train_model=config["train"],
                model=model,
                model_GNM=model_GNM, 
                text_encoder=text_encoder,
                model_nomad=model_nomad,
                noise_scheduler=noise_scheduler,                          
                optimizer=optimizer,
                lr_scheduler=scheduler,
                train_loader=train_loader,
                test_dataloaders=test_dataloaders,
                train_loader_sub=train_loader_sub,
                test_dataloaders_sub=test_dataloaders_sub,     
                train_loader_lan=train_loader_lelan,
                test_dataloaders_lan=test_loader_lelan,     
                train_loader_ft=train_loader_ft,
                test_dataloaders_ft=test_loader_ft,                                       
                transform=transform,
                #goal_mask_prob=config["goal_mask_prob"],
                epochs=config["epochs"],
                sacson=config["SACSoN"],
                device=device,
                batch_size=config["batch_size"],
                batch_size_test=config["eval_batch_size"], 
                len_traj_pred=config["len_traj_pred"],           
                #project_folder=config["project_folder"] + "_shadow_ft_scratch",
                project_folder=config["project_folder"] + "_ft_only",                
                print_log_freq=config["print_log_freq"],
                wandb_log_freq=config["wandb_log_freq"],
                image_log_freq=config["image_log_freq"],
                num_images_log=config["num_images_log"],
                current_epoch=current_epoch,
                alpha=float(config["alpha"]),
                use_wandb=config["use_wandb"],
                eval_fraction=config["eval_fraction"],
                eval_freq=config["eval_freq"],
            )        
        elif config["bdd"]:
            train_eval_loop_il_exaug_dist_gnm_gps_map2_lan_bdd(
                train_model=config["train"],
                model=model,
                model_GNM=model_GNM, 
                model_bdd=model_bdd, 
                text_encoder=text_encoder,
                model_nomad=model_nomad,
                noise_scheduler=noise_scheduler,                          
                optimizer=optimizer,
                lr_scheduler=scheduler,
                train_loader=train_loader,
                test_dataloaders=test_dataloaders,
                train_loader_sub=train_loader_sub,
                test_dataloaders_sub=test_dataloaders_sub,     
                train_loader_lan=train_loader_lelan,
                test_dataloaders_lan=test_loader_lelan,     
                train_loader_bdd=train_loader_bdd,
                test_dataloaders_bdd=test_loader_bdd,                                       
                transform=transform,
                #goal_mask_prob=config["goal_mask_prob"],
                epochs=config["epochs"],
                sacson=config["SACSoN"],
                device=device,
                batch_size=config["batch_size"],
                batch_size_test=config["eval_batch_size"], 
                len_traj_pred=config["len_traj_pred"],           
                #project_folder=config["project_folder"] + "_shadow_ft_scratch",
                #project_folder=config["project_folder"] + "_bdd2",       
                project_folder=config["project_folder"],
                print_log_freq=config["print_log_freq"],
                wandb_log_freq=config["wandb_log_freq"],
                image_log_freq=config["image_log_freq"],
                num_images_log=config["num_images_log"],
                current_epoch=current_epoch,
                alpha=float(config["alpha"]),
                use_wandb=config["use_wandb"],
                eval_fraction=config["eval_fraction"],
                eval_freq=config["eval_freq"],
            )                
        else:    
            train_eval_loop_il_exaug_dist_gnm_gps_map2_lan(
                train_model=config["train"],
                model=model,
                model_GNM=model_GNM, 
                text_encoder=text_encoder,
                model_nomad=model_nomad,
                noise_scheduler=noise_scheduler,                          
                optimizer=optimizer,
                lr_scheduler=scheduler,
                train_loader=train_loader,
                test_dataloaders=test_dataloaders,
                train_loader_sub=train_loader_sub,
                test_dataloaders_sub=test_dataloaders_sub,     
                train_loader_lan=train_loader_lelan,
                test_dataloaders_lan=test_loader_lelan,                       
                transform=transform,
                #goal_mask_prob=config["goal_mask_prob"],
                epochs=config["epochs"],
                sacson=config["SACSoN"],
                device=device,
                batch_size=config["batch_size"],
                batch_size_test=config["eval_batch_size"], 
                len_traj_pred=config["len_traj_pred"],           
                #project_folder=config["project_folder"] + "_shadow",
                project_folder=config["project_folder"],                
                print_log_freq=config["print_log_freq"],
                wandb_log_freq=config["wandb_log_freq"],
                image_log_freq=config["image_log_freq"],
                num_images_log=config["num_images_log"],
                current_epoch=current_epoch,
                alpha=float(config["alpha"]),
                use_wandb=config["use_wandb"],
                eval_fraction=config["eval_fraction"],
                eval_freq=config["eval_freq"],
                lan_solo=config["lan_solo"],
                image_solo=config["image_solo"],
                sate_solo=config["sate_solo"],
                no_sate=config["no_sate"],
            )                
    elif config["model_type"] == "annotate":
        train_eval_loop_annotate(
            train_model=config["train"],
            #model=model,
            #optimizer=optimizer,
            #lr_scheduler=scheduler,
            #noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            #transform=transform,
            #goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            batch_size=config["batch_size"],
            batch_size_test=config["eval_batch_size"], 
            #len_traj_pred=config["len_traj_pred"],           
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            #alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )                                                                             
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    #torch.multiprocessing.set_start_method("spawn")
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


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

    if config.get("lan_only_ft", False):
        # self-contained fine-tune: skip the frodobot project_folder / wandb-resume setup
        config.setdefault("project_folder", config.get("output_dir", "./logs_frodo_lan_ft"))
        if config.get("use_wandb", False):
            wandb.init(project=config["project_name"], settings=wandb.Settings(start_method="fork"))
            wandb.run.name = config["run_name"]
            wandb.config.update(config, allow_val_change=True)
        main(config)
        raise SystemExit(0)

    if config["keep_learning"]:
        if "load_run" in config: 
            config["project_folder"] = os.path.join(path_save_load, "logs", config["load_run"])
            load_run_dir = config["load_run"].split("/")
            config["run_name"] = load_run_dir[1]            
        else:
            config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
            config["project_folder"] = os.path.join(
                path_save_load, "logs", config["project_name"], config["run_name"]
            )
            os.makedirs(
                config[
                    "project_folder"
                ],  # should error if dir already exists to avoid overwriting and old project
            )            
    else:
        config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
        config["project_folder"] = os.path.join(
            path_save_load, "logs", config["project_name"], config["run_name"]
        )
        os.makedirs(
            config[
                "project_folder"
            ],  # should error if dir already exists to avoid overwriting and old project
        )
                
        
    if config["use_wandb"]:
        os.environ["WANDB_API_KEY"] = "f75d70faa46b5b328a3a7c85922ede0db348c2a6"
        wandb.login()
        if "load_run" in config and config["keep_learning"] and not config["aug_shadow"] and not config["ft_frod"]:
            api = wandb.Api()
            runs = api.runs(path=config["project_name"])
            print("config[project_name]", config["project_name"])
            print("runs", runs)
            
            for iw in runs:
                if iw.name == config["run_name"]:
                    id_number = iw.id
                    print("id_number", id_number)
            
            #id_number = "509dlfjn"
            try:
                wandb.init(
                    project=config["project_name"],
                    settings=wandb.Settings(start_method="fork"),
                    id=id_number,
                    resume="allow"
                )
                wandb.save(args.config, policy="now")  
            except:
                wandb.init(
                    project=config["project_name"],
                    settings=wandb.Settings(start_method="fork"),
                )            
                wandb.save(args.config, policy="now")
                wandb.run.name = config["run_name"]            
        else:
            wandb.init(
                project=config["project_name"],
                settings=wandb.Settings(start_method="fork"),
            )
            wandb.save(args.config, policy="now")  # save the config file
            #wandb.run.name = config["run_name"] + "_ft_only"
            #wandb.run.name = config["run_name"] + "_shadow_ft"
            #wandb.run.name = config["run_name"] + "_shadow_10k"            
            #wandb.run.name = config["run_name"] + "_bdd2"            
            wandb.run.name = config["run_name"]            
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config, allow_val_change=True)
    main(config)
