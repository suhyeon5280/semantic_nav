"""
=== EVAL TOOL 3/3: infer.py — deployment-style inference (NO dataset needed) ===
The three eval tools under train/eval/ each do a DIFFERENT job:
  - eval_compare.py   : QUANTITATIVE. Numeric metric table (base vs fine-tuned) on the
                        frodo_lan TEST split. Answers "how much better, in numbers?"
  - visualize_traj.py : QUALITATIVE. Overlays GT / base / fine-tuned paths on DATASET
                        samples (needs pickles+images). Answers "what do the paths look like?"
  - infer.py (THIS)   : DEPLOYMENT. Runs on ARBITRARY images + a text prompt; no labels /
                        no dataset format needed. Answers "what would the model do on THIS image?"

Give it 1+ images (a short context sequence, oldest->newest; the last is the current frame)
and a text instruction; it prints the predicted trajectory and saves a scene+trajectory PNG.

Only the observation images + the language prompt are used (goal mask 7 = language modality;
GPS / satellite-map / image-goal tokens are masked, so they are passed as dummy zeros).

Usage (run from train/):
    python eval/infer.py --prompt "go to the metal gate" --images path/to/cur.jpg
    python eval/infer.py --prompt "turn toward the white wall" \
        --images f_t-2.jpg f_t-1.jpg f_t.jpg              # oldest -> newest (last = current)
    python eval/infer.py --prompt "..." --images cur.jpg --ckpt logs_frodo_lan_ft/best.pth

Results are saved under train/eval/results/infer/<timestamp>/ (PNG + waypoints.txt).
"""
import argparse, yaml, os, sys, time
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import clip

# --- this file lives in train/eval/; make imports + relative paths behave as if run from train/ ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)
os.chdir(_TRAIN)
from vint_train.models.il.il import IL_gps_map_mask3_lan2

IMG = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _result_dir(tool):
    d = os.path.join(_HERE, "results", tool, time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(d, exist_ok=True)
    return d


def build_model(cfg):
    return IL_gps_map_mask3_lan2(
        context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"], learn_angle=cfg["learn_angle"],
        obs_encoder=cfg["obs_encoder"], obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"], mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"])


def load_ckpt(path, cfg, dev):
    m = build_model(cfg)
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[ckpt] {path}  (missing={len(miss)} unexpected={len(unexp)})")
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="language instruction")
    ap.add_argument("--images", nargs="+", required=True, help="image path(s), oldest->newest (last = current)")
    ap.add_argument("--ckpt", default="logs_frodo_lan_ft/best.pth")
    ap.add_argument("-c", "--config", default="config/frodo_lan_ft.yaml")
    ap.add_argument("--out", default=None, help="output PNG path (default: eval/results/infer/<timestamp>/infer.png)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/defaults.yaml")); cfg.update(yaml.safe_load(open(args.config)))
    # frodo_lan has no entry in data_config.yaml -> fall back to 0.125 (matches training normalization)
    mws = yaml.safe_load(open("vint_train/data/data_config.yaml")).get("frodo_lan", {}).get("metric_waypoint_spacing", 0.125)
    cs, H = cfg["context_size"], cfg["image_size"][0]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imgs = [TF.resize(TF.to_tensor(Image.open(p).convert("RGB")), (224, 224)) for p in args.images]
    cur = imgs[-1]
    # build a context of (cs+1) frames oldest->newest (pad by repeating the earliest if too few)
    seq = imgs[-(cs + 1):]
    while len(seq) < cs + 1:
        seq = [seq[0]] + seq
    obs = torch.cat([TF.resize(im, (H, H)) for im in seq]).unsqueeze(0).to(dev)   # [1, 3*(cs+1), H, W]

    ol = torch.split(obs, 3, dim=1); obs_map = ol[-1]
    obs_t = torch.cat([IMG(x) for x in ol], dim=1)
    z = torch.zeros(1, 3, H, H).to(dev)
    map_t = torch.cat((IMG(z), IMG(z), obs_map), axis=1)          # map token is masked under mask 7
    goal_img = IMG(z)                                             # image-goal token masked
    cur_t = IMG(TF.resize(cur, (224, 224)).unsqueeze(0).to(dev))  # feeds the FiLM language encoder
    goal_pose = torch.zeros(1, 4).to(dev)                         # gps token masked
    gm = torch.full((1,), 7, dtype=torch.long, device=dev)        # 7 = language-only

    txt, _ = clip.load(cfg["clip_type"]); txt.to(torch.float32).to(dev)
    with torch.no_grad():
        feat = txt.encode_text(clip.tokenize([args.prompt], truncate=True).to(dev))

    model = load_ckpt(args.ckpt, cfg, dev)
    with torch.no_grad():
        action, dist, _ = model(obs_t, goal_pose, map_t, goal_img, gm, feat, cur_t)
    traj = action[0].cpu().numpy()          # (8,4): (x=fwd, y=left, cos, sin) normalized
    xy_m = traj[:, :2] * mws                 # meters

    rdir = _result_dir("infer")
    out_png = args.out or os.path.join(rdir, "infer.png")

    lines = [f'prompt: "{args.prompt}"', f"ckpt: {args.ckpt}", "predicted waypoints (forward m, left m):"]
    for i, (f, l) in enumerate(xy_m):
        lines.append(f"  {i+1}: forward={f:+.2f}  left={l:+.2f}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(rdir, "waypoints.txt"), "w") as fh:
        fh.write(report + "\n")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=130)
    a0.imshow(cur.permute(1, 2, 0).numpy()); a0.set_axis_off(); a0.set_title(f'"{args.prompt}"', fontsize=11)
    lx = np.concatenate([[0], xy_m[:, 1]]); fy = np.concatenate([[0], xy_m[:, 0]])
    a1.plot(lx, fy, "-o", color="#4E79A7", lw=2, ms=4, label="predicted")
    a1.plot(0, 0, "ks", ms=8); a1.annotate("robot", (0, 0), textcoords="offset points", xytext=(4, -10), fontsize=8)
    a1.set_aspect("equal", "datalim"); a1.invert_xaxis()
    a1.axhline(0, color="gray", lw=.5, alpha=.3); a1.axvline(0, color="gray", lw=.5, alpha=.3)
    a1.set_xlabel("left (m)"); a1.set_ylabel("forward (m)"); a1.legend(); a1.set_title("predicted trajectory")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight")
    print("saved:", out_png, "| result dir:", rdir)


if __name__ == "__main__":
    main()
