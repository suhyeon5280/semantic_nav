"""
Visualize language-goal trajectories: scene image (+ object bbox + prompt) alongside a
bird's-eye plot of GT vs base vs fine-tuned predicted paths.

Self-contained: reads pickles/images directly, does NOT modify or depend on the dataset class.

Usage (from train/):
    python visualize_traj.py                                   # base vs logs_frodo_lan_ft/best.pth
    python visualize_traj.py --ft logs_frodo_lan_ft/2026_.../best.pth
    python visualize_traj.py --ft A/best.pth --ft B/best.pth --n 4 --out compare.png
    python visualize_traj.py --episode episode_0037 --n 6
"""
import argparse, os, glob, re, random, yaml
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import clip
from vint_train.models.il.il import IL_gps_map_mask3_lan2

IMG = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prompt_text(p):
    while not isinstance(p, str):
        p = p[0]
    return p


def is_surface(p, bl):
    return len(set(re.findall(r"[a-z]+", prompt_text(p).lower())) & bl) > 0


def load_224(path):
    return TF.resize(TF.to_tensor(Image.open(path).convert("RGB")), (224, 224))  # [3,224,224] in [0,1]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/frodo_lan_ft.yaml")
    ap.add_argument("--base", default=None, help="base ckpt (default: config load_edge_ckpt)")
    ap.add_argument("--ft", action="append", default=None, help="fine-tuned ckpt(s); repeatable")
    ap.add_argument("--episode", default=None, help="restrict samples to this episode folder")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_traj_scene.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/defaults.yaml")); cfg.update(yaml.safe_load(open(args.config)))
    root = cfg["datasets_lan"]["frodo_lan"]["pickle"]
    mws = yaml.safe_load(open("vint_train/data/data_config.yaml")).get("frodo_lan", {}).get("metric_waypoint_spacing", 0.12)
    bl = set(w.lower() for w in cfg.get("prompt_blocklist", []))
    cs = cfg["context_size"]; H = cfg["image_size"][0]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    base_path = args.base or cfg["load_edge_ckpt"]
    ft_paths = args.ft or ["logs_frodo_lan_ft/best.pth"]

    # ---- gather candidate (episode, frame_idx, obj) with a non-surface object ----
    eps = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "pickle_nomad")))
    if args.episode:
        eps = [args.episode]
    cands = []
    for ep in eps:
        pk_dir = os.path.join(root, ep, "pickle_nomad")
        im_dir = os.path.join(root, ep, "image")
        stems = sorted(f[:-4] for f in os.listdir(pk_dir) if f.endswith(".pkl"))
        idx = {s: k for k, s in enumerate(stems)}
        for s in stems:
            if not os.path.exists(os.path.join(im_dir, s + ".jpg")):
                continue
            objs = __import__("pickle").load(open(os.path.join(pk_dir, s + ".pkl"), "rb"))
            for oi, o in enumerate(objs):
                ps = o["prompt"]
                if bl and all(is_surface(ps[j], bl) for j in range(len(ps))):
                    continue
                cands.append((ep, im_dir, stems, idx[s], oi))
                break  # one object per frame is enough
    random.shuffle(cands)
    cands = cands[:args.n]
    print(f"{len(cands)} samples from {len(eps)} episode(s)")

    import pickle as pk
    scenes, bboxes, prompts, gts, objposes = [], [], [], [], []
    obs_b, goal_b, clg_b, gpos_b = [], [], [], []
    for ep, im_dir, stems, k, oi in cands:
        cur = load_224(os.path.join(im_dir, stems[k] + ".jpg"))
        ctx = [cur] + [load_224(os.path.join(im_dir, stems[max(0, k - h)] + ".jpg")) for h in range(1, cs + 1)]
        obs = torch.cat([TF.resize(im, (H, H)) for im in ctx[::-1]])          # oldest->newest, 3*(cs+1) ch
        o = pk.load(open(os.path.join(root, ep, "pickle_nomad", stems[k] + ".pkl"), "rb"))[oi]
        bb = np.asarray(o["bbox"]).reshape(-1).astype(int)
        pm = np.asarray(o["pose_median"]).reshape(-1).astype(np.float32)      # (fwd,left) m
        tr = np.asarray(o["nomad_traj_norm"], dtype=np.float32)               # (8,4)
        pr = next((prompt_text(o["prompt"][j]) for j in range(len(o["prompt"])) if not is_surface(o["prompt"][j], bl)),
                  prompt_text(o["prompt"][0]))
        scenes.append(cur); bboxes.append(bb); prompts.append(pr); gts.append(tr); objposes.append(pm)
        obs_b.append(obs); goal_b.append(TF.resize(cur, (H, H))); clg_b.append(cur); gpos_b.append(pm)

    obs_b = torch.stack(obs_b).to(dev)
    goal_b = torch.stack(goal_b).to(dev)
    clg_b = torch.stack(clg_b).to(dev)
    gpos = torch.tensor(np.stack(gpos_b)).to(dev)
    B = obs_b.shape[0]

    ol = torch.split(obs_b, 3, dim=1); obs_map = ol[-1]                        # raw current @H
    obs_t = torch.cat([IMG(x) for x in ol], dim=1)
    z = torch.zeros(B, 3, H, H).to(dev)
    mp = torch.cat((IMG(z), IMG(z), obs_map), axis=1)
    goal_t = IMG(goal_b); clg_t = IMG(clg_b)
    dis = torch.sqrt(gpos[:, 1:2] ** 2 + gpos[:, 0:1] ** 2) + 1e-6
    gpose = torch.cat((gpos[:, 1:2], -gpos[:, 0:1], gpos[:, 1:2] / dis, -gpos[:, 0:1] / dis), axis=1).float().to(dev)
    gm = torch.full((B,), 7, dtype=torch.long, device=dev)

    txt, _ = clip.load(cfg["clip_type"]); txt.to(torch.float32).to(dev)
    with torch.no_grad():
        feat = txt.encode_text(clip.tokenize(prompts, truncate=True).to(dev))

    runs = {"base": base_path}
    for i, p in enumerate(ft_paths):
        runs[f"ft{i+1}" if len(ft_paths) > 1 else "fine-tuned"] = p
    preds = {}
    for name, p in runs.items():
        m = load_ckpt(p, cfg, dev)
        with torch.no_grad():
            a, _, _ = m(obs_t, gpose, mp, goal_t, gm, feat, clg_t)
        preds[name] = a.cpu().numpy()
        print(f"loaded {name}: {p}")

    # ---- plot: N rows x 2 cols (scene | trajectory) ----
    colors = ["#9aa0a6", "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    fig, axes = plt.subplots(B, 2, figsize=(9, 3.6 * B), dpi=130, squeeze=False)
    for r in range(B):
        axs, axt = axes[r, 0], axes[r, 1]
        # scene + bbox
        axs.imshow(scenes[r].permute(1, 2, 0).numpy()); axs.set_axis_off()
        t, b, l, rr = bboxes[r][:4]
        axs.add_patch(Rectangle((l, t), rr - l, b - t, fill=False, edgecolor="red", lw=2))
        axs.set_title(f'"{prompts[r]}"', fontsize=10)
        # trajectory
        def path(tr):
            return np.concatenate([[0], tr[:, 1] * mws]), np.concatenate([[0], tr[:, 0] * mws])  # left, fwd
        lx, fy = path(gts[r]); axt.plot(lx, fy, "k--o", lw=2, ms=3, label="GT", zorder=5)
        for i, (name, pr) in enumerate(preds.items()):
            lx, fy = path(pr[r]); axt.plot(lx, fy, "-o", color=colors[i % len(colors)], lw=1.8, ms=3, label=name)
        axt.plot(0, 0, "ks", ms=8); axt.plot(objposes[r][1], objposes[r][0], "r*", ms=16, label="object", zorder=6)
        axt.set_aspect("equal", "datalim"); axt.invert_xaxis()
        axt.axhline(0, color="gray", lw=.5, alpha=.3); axt.axvline(0, color="gray", lw=.5, alpha=.3)
        axt.set_xlabel("left (m)", fontsize=8); axt.set_ylabel("forward (m)", fontsize=8); axt.tick_params(labelsize=7)
        if r == 0:
            axt.legend(fontsize=8, loc="best")
    fig.suptitle("Scene + object (red box) vs predicted trajectories (language goal)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
