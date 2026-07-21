"""
=== EVAL TOOL 5: grounding_test.py — does the model select the PROMPTED object? ===
A rigorous proof of language grounding (vs base), WITH a ground truth and a chance baseline
(unlike "it went toward the object", which has neither).

Idea: many frames contain >=2 annotated objects at KNOWN robot-relative positions
(pose_median). Fix the observation; prompt with object A's description, then B's. If the
model grounds language, prompting the more-LEFT object should give a more-LEFT endpoint than
prompting the more-RIGHT object.

PRIMARY METRIC — target-selection accuracy:
    over all object pairs (A,B) in a frame whose lateral positions differ by > --min-sep,
    correct if sign(endpoint_left(A) - endpoint_left(B)) == sign(posL(A) - posL(B)).
    accuracy = correct / total.  Chance = 50%.  Reported for base vs fine-tuned.
SECONDARY — Pearson r between object lateral position and predicted endpoint lateral
    (pooled over all objects; higher = stronger grounding).

Everything else (observation, mask=7, gps/map/image-goal tokens) is held constant, so any
signal is language-driven. The noun (object description) is the only thing that varies, so
this measures OBJECT grounding — NOT spatial words like "left"/"right" (see direction_test.py).

Usage (run from train/):
    python eval/grounding_test.py --ft logs_frodo_lan_ft/<run>/best.pth --n 25
Results -> train/eval/results/grounding_test/<timestamp>/ (scatter.png + summary.txt)
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
from vint_train.models.il.il import IL_gps_map_mask3_lan2, clip_token_features

IMG = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _result_dir(tool):
    d = os.path.join(_HERE, "results", tool, time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(d, exist_ok=True)
    return d


def load224(p):
    return TF.resize(TF.to_tensor(Image.open(p).convert("RGB")), (224, 224))


def ptxt(p):
    while not isinstance(p, str):
        p = p[0]
    return p


def is_surface(p, bl):
    return len(set(re.findall(r"[a-z]+", ptxt(p).lower())) & bl) > 0


def build(cfg):
    return IL_gps_map_mask3_lan2(context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"], learn_angle=cfg["learn_angle"],
        obs_encoder=cfg["obs_encoder"], obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"], mha_num_attention_layers=cfg["mha_num_attention_layers"], mha_ff_dim_factor=cfg["mha_ff_dim_factor"])


def load_ckpt(p, cfg, dev, use_lgx=False):
    m = build(cfg); sd = torch.load(p, map_location="cpu"); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}; m.load_state_dict(sd, strict=False)
    m.use_lgx = use_lgx
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/frodo_lan_ft.yaml")
    ap.add_argument("--base", default=None)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--n", type=int, default=25, help="number of multi-object frames to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--min-sep", type=float, default=0.3, help="min lateral separation (m) for a scorable object pair")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/defaults.yaml")); cfg.update(yaml.safe_load(open(args.config)))
    root = cfg["datasets_lan"]["frodo_lan"]["pickle"]
    mws = 0.125; cs = cfg["context_size"]; H = cfg["image_size"][0]
    bl = set(w.lower() for w in cfg.get("prompt_blocklist", []))
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    base_path = args.base or cfg["load_edge_ckpt"]

    # ---- split-aware frame list (same logic as the other eval tools) ----
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

    import pickle as pk
    # ---- collect frames that have >=2 non-surface objects with distinct lateral positions ----
    scenes = []  # each: dict(obs, clg, objs=[(prompt, posL)])
    cand = [(e, s) for (e, s) in all_frames if (e, s) in allowed and epstems[e].index(s) >= cs]
    random.shuffle(cand)
    for ep, st in cand:
        if len(scenes) >= args.n:
            break
        objs = pk.load(open(os.path.join(root, ep, "pickle_nomad", st + ".pkl"), "rb"))
        items = []
        for o in objs:
            prs = o["prompt"]
            pref = [j for j in range(len(prs)) if not is_surface(prs[j], bl)]
            if not pref:
                continue
            pm = np.asarray(o["pose_median"]).reshape(-1).astype(np.float32)
            items.append((ptxt(prs[pref[0]]), float(pm[1])))   # (prompt, lateral=left)
        # need >=2 objects separated laterally
        if len(items) < 2 or (max(i[1] for i in items) - min(i[1] for i in items)) < args.min_sep:
            continue
        k = epstems[ep].index(st); imd = os.path.join(root, ep, "image")
        cur = load224(os.path.join(imd, st + ".jpg"))
        ctx = [cur] + [load224(os.path.join(imd, epstems[ep][max(0, k - h)] + ".jpg")) for h in range(1, cs + 1)]
        obs = torch.cat([TF.resize(im, (H, H)) for im in ctx[::-1]])
        scenes.append(dict(obs=obs, clg=cur, objs=items))
    print(f"[grounding] split={args.split} | {len(scenes)} multi-object frames (>=2 objs, lateral sep>={args.min_sep}m)")

    txt, _ = clip.load(cfg["clip_type"]); txt.to(torch.float32).to(dev)
    models = {"base": load_ckpt(base_path, cfg, dev, use_lgx=False),
              "fine-tuned": load_ckpt(args.ft, cfg, dev, use_lgx=bool(cfg.get("use_lgx", False)))}

    def endpoints_left(model, obs, clg, prompts):
        P = len(prompts)
        obs_b = obs.unsqueeze(0).repeat(P, 1, 1, 1).to(dev)
        ol = torch.split(obs_b, 3, dim=1); obs_map = ol[-1]
        obs_t = torch.cat([IMG(x) for x in ol], dim=1); z = torch.zeros(P, 3, H, H).to(dev)
        mp = torch.cat((IMG(z), IMG(z), obs_map), 1); clg_t = IMG(clg.unsqueeze(0).repeat(P, 1, 1, 1).to(dev))
        gp = torch.zeros(P, 4).to(dev); gm = torch.full((P,), 7, dtype=torch.long, device=dev)
        with torch.no_grad():
            tok = clip.tokenize(prompts, truncate=True).to(dev)
            feat = txt.encode_text(tok)
            tt, tv = (clip_token_features(txt, tok) if getattr(model, "use_lgx", False) else (None, None))
            a, _, _ = model(obs_t, gp, mp, IMG(z), gm, feat, clg_t, tt, tv)
        return (a[:, -1, 1].cpu().numpy() * mws)   # endpoint lateral (m), + = left

    results = {}
    for name, m in models.items():
        correct = total = 0
        xs, ys = [], []   # (object true lateral, predicted endpoint lateral) for correlation
        for sc in scenes:
            prompts = [o[0] for o in sc["objs"]]; posL = [o[1] for o in sc["objs"]]
            eL = endpoints_left(m, sc["obs"], sc["clg"], prompts)
            xs += posL; ys += list(eL)
            for i in range(len(prompts)):
                for j in range(i + 1, len(prompts)):
                    if abs(posL[i] - posL[j]) < args.min_sep:
                        continue
                    total += 1
                    if np.sign(eL[i] - eL[j]) == np.sign(posL[i] - posL[j]):
                        correct += 1
        acc = correct / total if total else float("nan")
        r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
        results[name] = dict(acc=acc, n=total, r=r, xs=xs, ys=ys)

    # ---- report ----
    rdir = _result_dir("grounding_test")
    lines = ["=== Language grounding: target-selection accuracy (prompt the object, does the",
             "    endpoint follow the PROMPTED object's real lateral position?) ===",
             f"split={args.split} | frames={len(scenes)} | chance=50%", ""]
    lines.append(f"{'model':12s} {'target-sel acc':>15s} {'pairs':>7s} {'pearson r(pos,endpoint)':>25s}")
    for name in models:
        rr = results[name]
        lines.append(f"{name:12s} {rr['acc']*100:14.1f}% {rr['n']:7d} {rr['r']:25.3f}")
    b, f = results["base"], results["fine-tuned"]
    lines += ["",
              f"=> fine-tuned selects the prompted object {f['acc']*100:.0f}% of the time "
              f"(base {b['acc']*100:.0f}%, chance 50%).",
              f"   correlation(object position, endpoint): base {b['r']:.2f} -> fine-tuned {f['r']:.2f} "
              "(higher = the trajectory tracks WHICH object was named)."]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(rdir, "summary.txt"), "w") as fh:
        fh.write(report + "\n")

    # ---- scatter: object true lateral vs predicted endpoint lateral ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=130)
    for ax, name in zip(axes, models):
        rr = results[name]
        ax.scatter(rr["xs"], rr["ys"], s=18, alpha=0.6, color="#4E79A7")
        lim = max(0.5, np.abs(rr["xs"] + rr["ys"]).max() if rr["xs"] else 0.5)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5, label="perfect grounding")
        ax.axhline(0, color="gray", lw=.4); ax.axvline(0, color="gray", lw=.4)
        ax.set_xlabel("object true lateral position (m, + = left)")
        ax.set_ylabel("predicted endpoint lateral (m, + = left)")
        ax.set_title(f"{name}  (acc={rr['acc']*100:.0f}%, r={rr['r']:.2f})")
        ax.legend(fontsize=8)
    fig.suptitle("Does the predicted trajectory track the PROMPTED object's position? (positive slope = grounding)")
    fig.tight_layout()
    fig.savefig(os.path.join(rdir, "scatter.png"), bbox_inches="tight")
    print("saved:", os.path.join(rdir, "scatter.png"), "| result dir:", rdir)


if __name__ == "__main__":
    main()
