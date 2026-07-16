"""
=== EVAL TOOL 1/3: eval_compare.py — quantitative metrics on the TEST split ===
The three eval tools under train/eval/ each do a DIFFERENT job:
  - eval_compare.py (THIS) : QUANTITATIVE. Runs base AND a fine-tuned checkpoint over the
                        frodo_lan TEST split and prints a side-by-side metric table
                        (action_mse, waypoint/endpoint/object error in meters, heading_cos)
                        plus an image-goal base-divergence number. Answers "how much better?"
  - visualize_traj.py : QUALITATIVE. Overlays GT / base / fine-tuned paths on dataset samples.
  - infer.py          : DEPLOYMENT. Runs on ARBITRARY images + a prompt; no dataset needed.

Reports, on the held-out test split:
  * language-goal (mask 7) metrics for BOTH models, side by side, with delta
  * a basic-driving regression number: image-goal (mask 6) divergence between the two models
No training data / teachers needed. Neither checkpoint file is modified.

Usage (run from train/):
    python eval/eval_compare.py                                 # base vs logs_frodo_lan_ft/best.pth
    python eval/eval_compare.py --ft ./logs_frodo_lan_ft/2026_.../best.pth
    python eval/eval_compare.py -c config/frodo_lan_ft.yaml --base ./omnivla-edge.pth --ft ...

Results (the printed table) are also saved to train/eval/results/eval_compare/<timestamp>/metrics.txt .
"""
import argparse
import os
import sys
import time
import yaml

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import clip

# --- this file lives in train/eval/; make imports + relative paths behave as if run from train/ ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)
os.chdir(_TRAIN)
from vint_train.models.il.il import IL_gps_map_mask3_lan2
from vint_train.data.lelan_dataset import LeLaN_Dataset_multi
from vint_train.training.train_utils import eval_metrics_lan, evaluate_lan_only_ft


def _result_dir(tool):
    d = os.path.join(_HERE, "results", tool, time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(d, exist_ok=True)
    return d


def build_model(c):
    return IL_gps_map_mask3_lan2(
        context_size=c["context_size"], len_traj_pred=c["len_traj_pred"], learn_angle=c["learn_angle"],
        obs_encoder=c["obs_encoder"], obs_encoding_size=c["obs_encoding_size"], late_fusion=c["late_fusion"],
        mha_num_attention_heads=c["mha_num_attention_heads"],
        mha_num_attention_layers=c["mha_num_attention_layers"], mha_ff_dim_factor=c["mha_ff_dim_factor"],
    )


def load_into(m, path, device):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  [{os.path.basename(path)}] loaded (missing={len(missing)} unexpected={len(unexpected)})")
    return m.to(device).eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/frodo_lan_ft.yaml")
    ap.add_argument("--base", default=None, help="base checkpoint (default: config load_edge_ckpt)")
    ap.add_argument("--ft", default="./logs_frodo_lan_ft/best.pth", help="fine-tuned checkpoint")
    args = ap.parse_args()

    with open("config/defaults.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(args.config) as f:
        cfg.update(yaml.safe_load(f))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dc = cfg["datasets_lan"]["frodo_lan"]
    # match training: same split strategy + prompt filter
    LeLaN_Dataset_multi.split_by_episode = bool(cfg.get("split_by_episode", False))
    LeLaN_Dataset_multi.test_episodes = cfg.get("test_episodes", None)
    _default_bl = ["asphalt", "road", "roads", "roadway", "roadways", "pavement", "paved", "tarmac",
                   "ground", "floor", "surface", "sidewalk", "gravel", "dirt", "lane", "lanes",
                   "path", "pathway", "drop", "mud", "cobblestone", "area"]
    _blocklist = set(w.lower() for w in cfg.get("prompt_blocklist", _default_bl))
    print(f"[eval] split={'episode' if LeLaN_Dataset_multi.split_by_episode else 'index'} "
          f"| prompt-filter {'ON' if _blocklist else 'off'}")
    test_ds = LeLaN_Dataset_multi(
        data_split_folder=dc["test"], dataset_name="frodo_lan", image_size=cfg["image_size"],
        waypoint_spacing=dc.get("waypoint_spacing", 1), len_traj_pred=cfg["len_traj_pred"],
        learn_angle=cfg["learn_angle"], context_size=cfg["context_size"], data_split_type="test",
        data_image_folder=dc["image"], data_pickle_folder=dc["pickle"], lan_solo=True,
        context_type=cfg.get("context_type", "temporal"), normalize=cfg["normalize"],
        backside=dc.get("backside", False), aug_seq=dc.get("aug_seq", False), only_front=dc.get("only_front", True),
    )
    test_ds.prompt_blocklist = _blocklist
    loader = DataLoader(test_ds, batch_size=cfg.get("eval_batch_size", cfg["batch_size"]),
                        shuffle=False, num_workers=cfg["num_workers"], drop_last=False)

    text_encoder, _ = clip.load(cfg["clip_type"])
    text_encoder.to(torch.float32).to(device)

    base_path = args.base or cfg["load_edge_ckpt"]
    print("Loading base       :", base_path)
    base = load_into(build_model(cfg), base_path, device)
    print("Loading fine-tuned :", args.ft)
    ft = load_into(build_model(cfg), args.ft, device)

    # tee: everything appended to `report` is both printed and written to metrics.txt
    report = []
    def out(s=""):
        print(s)
        report.append(str(s))

    out(f"base       : {base_path}")
    out(f"fine-tuned : {args.ft}")
    out("\n=== Language-goal (mask 7) metrics on frodo_lan TEST split ===")
    mb = eval_metrics_lan(base, text_encoder, loader, transform, device, goal_mask_value=7)
    mf = eval_metrics_lan(ft, text_encoder, loader, transform, device, goal_mask_value=7)
    better = {
        "action_mse": "lower", "waypoint_err_m": "lower", "endpoint_err_m": "lower",
        "object_err_m": "lower", "heading_cos": "higher",
    }
    out(f"{'metric':16s} {'base':>12s} {'finetuned':>12s} {'delta':>12s}   better")
    for k in ["action_mse", "waypoint_err_m", "endpoint_err_m", "object_err_m", "heading_cos"]:
        delta = mf[k] - mb[k]
        improved = (delta < 0) if better[k] == "lower" else (delta > 0)
        mark = "  <-- improved" if improved else ""
        out(f"{k:16s} {mb[k]:12.4f} {mf[k]:12.4f} {delta:+12.4f}   ({better[k]}){mark}")
    out(f"(test samples = {mb['n']})")

    out("\n=== Basic-driving regression: image-goal (mask 6) divergence, base vs fine-tuned ===")
    div = evaluate_lan_only_ft(ft, base, text_encoder, loader, transform, device)
    out(f"  base_divergence(image-goal) = {div['base_divergence_imagegoal']:.4f}")
    out("  (~0 = image-goal driving preserved; large = fine-tune shifted basic driving)")

    rdir = _result_dir("eval_compare")
    with open(os.path.join(rdir, "metrics.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print("\nresult dir:", rdir)
