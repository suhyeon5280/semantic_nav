"""
=== EVAL TOOL 4: prompt_sensitivity.py — does the model actually LISTEN to language? ===
Experiment: fix the observation (a real 6-frame context from the TEST split) and vary ONLY
the language prompt. Everything else (images, mask=7, gps/map/image-goal tokens) is held
constant, so any change in the predicted trajectory is caused PURELY by the text.

  - A model that GROUNDS language -> its trajectories fan out as the prompt changes.
  - A model that (nearly) IGNORES language -> all trajectories collapse onto one line.

Quantitative metric: "language sensitivity" = mean pairwise distance (meters) between the
per-prompt trajectory ENDPOINTS. Higher = more language-sensitive. Reported for base vs FT.

Usage (run from train/):
    python eval/prompt_sensitivity.py --ft logs_frodo_lan_ft/<run>/best.pth --n 4
Results -> train/eval/results/prompt_sensitivity/<timestamp>/ (sensitivity.png + summary.txt)
"""
import argparse, os, sys, re, random, time, yaml
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import clip

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
if _TRAIN not in sys.path:
    sys.path.insert(0, _TRAIN)
os.chdir(_TRAIN)
from vint_train.models.il.il import IL_gps_map_mask3_lan2

IMG = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# diverse prompts that SHOULD steer differently if the model grounds language
PROMPTS = [
    "go to the wall on the right",
    "go to the car on the left",
    "follow the road straight ahead",
    "turn toward the building",
    "stop",
]
PCOLORS = ["#4E79A7", "#E15759", "#59A14F", "#B07AA1", "#F28E2B"]


def _result_dir(tool):
    d = os.path.join(_HERE, "results", tool, time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(d, exist_ok=True)
    return d


def load_224(path):
    return TF.resize(TF.to_tensor(Image.open(path).convert("RGB")), (224, 224))


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
    m.load_state_dict(sd, strict=False)
    return m.to(dev).eval()


def mean_pairwise_endpoint_dist(endpoints):
    # endpoints: (P, 2) meters -> mean of all pairwise Euclidean distances
    P = len(endpoints)
    if P < 2:
        return 0.0
    d, cnt = 0.0, 0
    for i in range(P):
        for j in range(i + 1, P):
            d += float(np.linalg.norm(endpoints[i] - endpoints[j])); cnt += 1
    return d / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/frodo_lan_ft.yaml")
    ap.add_argument("--base", default=None, help="base ckpt (default: config load_edge_ckpt)")
    ap.add_argument("--ft", required=True, help="fine-tuned ckpt to contrast against base")
    ap.add_argument("--n", type=int, default=4, help="number of test scenes to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/defaults.yaml")); cfg.update(yaml.safe_load(open(args.config)))
    root = cfg["datasets_lan"]["frodo_lan"]["pickle"]
    mws = yaml.safe_load(open("vint_train/data/data_config.yaml")).get("frodo_lan", {}).get("metric_waypoint_spacing", 0.125)
    cs = cfg["context_size"]; H = cfg["image_size"][0]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    base_path = args.base or cfg["load_edge_ckpt"]

    # ---- split-aware scene sampling (same logic as visualize_traj) ----
    split_by_episode = bool(cfg.get("split_by_episode", False))
    all_eps = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "pickle_nomad")))
    all_frames = []
    epstems = {}
    for ep in all_eps:
        im_dir = os.path.join(root, ep, "image")
        stems = sorted(f[:-4] for f in os.listdir(os.path.join(root, ep, "pickle_nomad")) if f.endswith(".pkl"))
        epstems[ep] = [s for s in stems if os.path.exists(os.path.join(im_dir, s + ".jpg"))]
        for st in epstems[ep]:
            all_frames.append((ep, st))
    if args.split == "all":
        allowed = set(all_frames)
    elif split_by_episode:
        forced = cfg.get("test_episodes")
        if forced:
            test_eps = set(forced)
        else:
            n_test_ep = max(1, int(round(len(all_eps) * 0.1)))
            test_eps = set(all_eps[-n_test_ep:])
        allowed = set((ep, st) for (ep, st) in all_frames if (ep in test_eps) == (args.split == "test"))
    else:
        thres = int(len(all_frames) * 0.9)
        allowed = set(all_frames[thres:] if args.split == "test" else all_frames[:thres])
    # need >=cs preceding frames in the same episode for a real context
    cand = [(ep, st) for (ep, st) in all_frames if (ep, st) in allowed and epstems[ep].index(st) >= cs]
    random.shuffle(cand)
    cand = cand[:args.n]
    print(f"[split] {args.split} | {len(allowed)}/{len(all_frames)} eligible | {len(cand)} scenes sampled")

    # ---- build a batched observation per scene ----
    scenes_img, obs_list, clg_list = [], [], []
    for ep, st in cand:
        stems = epstems[ep]; k = stems.index(st)
        cur = load_224(os.path.join(root, ep, "image", st + ".jpg"))
        ctx = [cur] + [load_224(os.path.join(root, ep, "image", stems[max(0, k - h)] + ".jpg")) for h in range(1, cs + 1)]
        obs = torch.cat([TF.resize(im, (H, H)) for im in ctx[::-1]])  # oldest->newest
        scenes_img.append(cur); obs_list.append(obs); clg_list.append(cur)

    # ---- CLIP text features for each prompt (shared across scenes/models) ----
    txt, _ = clip.load(cfg["clip_type"]); txt.to(torch.float32).to(dev)
    with torch.no_grad():
        feats = txt.encode_text(clip.tokenize(PROMPTS, truncate=True).to(dev))  # (P, D)
    P = len(PROMPTS)

    models = {"base": load_ckpt(base_path, cfg, dev),
              "fine-tuned": load_ckpt(args.ft, cfg, dev)}

    # preds[model][scene] = (P, 8, 2) meters ; sens[model] = list of per-scene sensitivities
    preds = {m: [] for m in models}
    sens = {m: [] for m in models}
    for si in range(len(cand)):
        obs_b = obs_list[si].unsqueeze(0).repeat(P, 1, 1, 1).to(dev)
        ol = torch.split(obs_b, 3, dim=1); obs_map = ol[-1]
        obs_t = torch.cat([IMG(x) for x in ol], dim=1)
        z = torch.zeros(P, 3, H, H).to(dev)
        mp = torch.cat((IMG(z), IMG(z), obs_map), axis=1)
        goal_t = IMG(z)                                  # image-goal token (masked under mask 7)
        clg_t = IMG(clg_list[si].unsqueeze(0).repeat(P, 1, 1, 1).to(dev))
        gpose = torch.zeros(P, 4).to(dev)                # gps token (masked under mask 7)
        gm = torch.full((P,), 7, dtype=torch.long, device=dev)
        for name, m in models.items():
            with torch.no_grad():
                a, _, _ = m(obs_t, gpose, mp, goal_t, gm, feats, clg_t)
            xy = a[:, :, :2].cpu().numpy() * mws          # (P, 8, 2) meters
            preds[name].append(xy)
            sens[name].append(mean_pairwise_endpoint_dist(xy[:, -1, :]))

    # ---- report ----
    rdir = _result_dir("prompt_sensitivity")
    lines = ["=== Language sensitivity (fixed observation, prompt varied) ===",
             f"prompts ({P}): " + " | ".join(PROMPTS),
             f"base       : {base_path}",
             f"fine-tuned : {args.ft}",
             f"scenes     : {len(cand)} (split={args.split})", ""]
    lines.append(f"{'model':12s} {'mean endpoint spread (m)':>26s}   per-scene")
    for name in models:
        arr = np.array(sens[name])
        lines.append(f"{name:12s} {arr.mean():26.3f}   {np.round(arr,3).tolist()}")
    fold = (np.array(sens['fine-tuned']).mean() + 1e-9) / (np.array(sens['base']).mean() + 1e-9)
    lines.append("")
    lines.append(f"=> fine-tuned is {fold:.1f}x more language-sensitive than base "
                 f"(larger endpoint spread across prompts = listens to language more).")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(rdir, "summary.txt"), "w") as fh:
        fh.write(report + "\n")

    # ---- plot: rows = scenes, cols = [scene | base fan | fine-tuned fan] ----
    N = len(cand)
    fig, axes = plt.subplots(N, 3, figsize=(13, 3.6 * N), dpi=130, squeeze=False)
    for si in range(N):
        axes[si, 0].imshow(scenes_img[si].permute(1, 2, 0).numpy()); axes[si, 0].set_axis_off()
        axes[si, 0].set_title(f"scene {si+1} (obs fixed)", fontsize=10)
        for ci, name in enumerate(models):
            ax = axes[si, ci + 1]
            xy = preds[name][si]  # (P,8,2)
            for p in range(P):
                fwd = np.concatenate([[0], xy[p, :, 0]]); left = np.concatenate([[0], xy[p, :, 1]])
                ax.plot(left, fwd, "-o", color=PCOLORS[p % len(PCOLORS)], lw=1.6, ms=3,
                        label=PROMPTS[p] if si == 0 else None)
            ax.plot(0, 0, "ks", ms=7)
            ax.set_aspect("equal", "datalim"); ax.invert_xaxis()
            ax.axhline(0, color="gray", lw=.5, alpha=.3); ax.axvline(0, color="gray", lw=.5, alpha=.3)
            ax.set_xlabel("left (m)", fontsize=8); ax.set_ylabel("forward (m)", fontsize=8); ax.tick_params(labelsize=7)
            ax.set_title(f"{name}  (spread={sens[name][si]:.3f} m)", fontsize=10)
            if si == 0 and ci == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("Same observation, different prompt -> how much does the trajectory move? "
                 "(fan-out = listens to language)", fontsize=12)
    fig.tight_layout()
    out_png = args.out or os.path.join(rdir, "sensitivity.png")
    fig.savefig(out_png, bbox_inches="tight")
    print("saved:", out_png, "| result dir:", rdir)


if __name__ == "__main__":
    main()
