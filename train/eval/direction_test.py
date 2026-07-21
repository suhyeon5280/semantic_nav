"""
=== EVAL TOOL 6: direction_test.py — CONTROL: does the model use spatial words? ===
Minimal-pair probe: hold the NOUN constant and flip ONLY the direction word
("go to the wall on the LEFT" vs "...on the RIGHT"), on a fixed observation.

If the model grounds spatial words, the "left" prompt should steer more-left than the
"right" prompt: delta = endpoint_left(LEFT) - endpoint_left(RIGHT) > 0.
If delta ~= 0, the model IGNORES left/right and keys only on the object noun + scene.

This is the control that keeps grounding_test.py honest: grounding_test shows the model
selects the prompted OBJECT; this shows it does NOT parse spatial directions (expected, since
the training prompts are object noun-phrases with almost no "left/right/go/turn" words).

Usage (run from train/):
    python eval/direction_test.py --ft logs_frodo_lan_ft/<run>/best.pth --n 4
Results -> train/eval/results/direction_test/<timestamp>/summary.txt
"""
import argparse, os, sys, random, time, yaml
import numpy as np
from PIL import Image
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
NOUNS = ["wall", "car", "building", "fence"]


def _result_dir(tool):
    d = os.path.join(_HERE, "results", tool, time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(d, exist_ok=True)
    return d


def load224(p):
    return TF.resize(TF.to_tensor(Image.open(p).convert("RGB")), (224, 224))


def build(cfg):
    return IL_gps_map_mask3_lan2(context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"], learn_angle=cfg["learn_angle"],
        obs_encoder=cfg["obs_encoder"], obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"], mha_num_attention_layers=cfg["mha_num_attention_layers"], mha_ff_dim_factor=cfg["mha_ff_dim_factor"])


def load_ckpt(p, cfg, dev):
    m = build(cfg); sd = torch.load(p, map_location="cpu"); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}; m.load_state_dict(sd, strict=False)
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/frodo_lan_ft.yaml")
    ap.add_argument("--base", default=None)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--n", type=int, default=4, help="number of fixed scenes to average over")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/defaults.yaml")); cfg.update(yaml.safe_load(open(args.config)))
    root = cfg["datasets_lan"]["frodo_lan"]["pickle"]
    mws = 0.125; cs = cfg["context_size"]; H = cfg["image_size"][0]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    base_path = args.base or cfg["load_edge_ckpt"]

    # sample fixed scenes from the chosen split
    split_by_episode = bool(cfg.get("split_by_episode", False))
    all_eps = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "pickle_nomad")))
    epstems, all_frames = {}, []
    for ep in all_eps:
        imd = os.path.join(root, ep, "image")
        epstems[ep] = [s for s in sorted(f[:-4] for f in os.listdir(os.path.join(root, ep, "pickle_nomad")) if f.endswith(".pkl"))
                       if os.path.exists(os.path.join(imd, s + ".jpg"))]
        all_frames += [(ep, s) for s in epstems[ep]]
    if args.split == "all":
        allowed = set(all_frames)
    elif split_by_episode:
        forced = cfg.get("test_episodes")
        test_eps = set(forced) if forced else set(all_eps[-max(1, round(len(all_eps) * 0.1)):])
        allowed = set((e, s) for (e, s) in all_frames if (e in test_eps) == (args.split == "test"))
    else:
        thr = int(len(all_frames) * 0.9)
        allowed = set(all_frames[thr:] if args.split == "test" else all_frames[:thr])
    cand = [(e, s) for (e, s) in all_frames if (e, s) in allowed and epstems[e].index(s) >= cs]
    random.shuffle(cand); cand = cand[:args.n]

    obs_list, clg_list = [], []
    for ep, st in cand:
        k = epstems[ep].index(st); imd = os.path.join(root, ep, "image"); cur = load224(os.path.join(imd, st + ".jpg"))
        ctx = [cur] + [load224(os.path.join(imd, epstems[ep][max(0, k - h)] + ".jpg")) for h in range(1, cs + 1)]
        obs_list.append(torch.cat([TF.resize(im, (H, H)) for im in ctx[::-1]])); clg_list.append(cur)

    txt, _ = clip.load(cfg["clip_type"]); txt.to(torch.float32).to(dev)
    models = {"base": load_ckpt(base_path, cfg, dev), "fine-tuned": load_ckpt(args.ft, cfg, dev)}

    def endpoint_left(model, obs, clg, prompt):
        obs_b = obs.unsqueeze(0).to(dev); ol = torch.split(obs_b, 3, dim=1); obs_map = ol[-1]
        obs_t = torch.cat([IMG(x) for x in ol], dim=1); z = torch.zeros(1, 3, H, H).to(dev)
        mp = torch.cat((IMG(z), IMG(z), obs_map), 1); clg_t = IMG(clg.unsqueeze(0).to(dev))
        gp = torch.zeros(1, 4).to(dev); gm = torch.full((1,), 7, dtype=torch.long, device=dev)
        with torch.no_grad():
            feat = txt.encode_text(clip.tokenize([prompt], truncate=True).to(dev))
            a, _, _ = model(obs_t, gp, mp, IMG(z), gm, feat, clg_t)
        return float(a[0, -1, 1].cpu()) * mws   # + = left, - = right

    rdir = _result_dir("direction_test")
    lines = ["=== Direction control: SAME noun, flip only left<->right (fixed observation) ===",
             f"split={args.split} | scenes={len(cand)}",
             "delta = endpoint_left(LEFT prompt) - endpoint_left(RIGHT prompt); >0 => understands direction", "",
             f"{'model':12s} {'noun':9s} {'LEFT':>8s} {'RIGHT':>8s} {'delta':>8s}  verdict"]
    for name, m in models.items():
        ds = []
        for noun in NOUNS:
            L = float(np.mean([endpoint_left(m, obs_list[i], clg_list[i], f"go to the {noun} on the left") for i in range(len(cand))]))
            R = float(np.mean([endpoint_left(m, obs_list[i], clg_list[i], f"go to the {noun} on the right") for i in range(len(cand))]))
            d = L - R; ds.append(d)
            v = "L>R (understands)" if d > 0.02 else ("R>L reversed" if d < -0.02 else "~0 (no direction effect)")
            lines.append(f"{name:12s} {noun:9s} {L:8.3f} {R:8.3f} {d:8.3f}  {v}")
        lines.append(f"{name:12s} {'MEAN':9s} {'':8s} {'':8s} {np.mean(ds):8.3f}")
        lines.append("-" * 60)
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(rdir, "summary.txt"), "w") as fh:
        fh.write(report + "\n")
    print("result dir:", rdir)


if __name__ == "__main__":
    main()
