#!/usr/bin/env python
"""Integrity check for the LeLaN-format dataset before training.
Verifies every episode's jpg (openable, not truncated) and pkl (loadable, expected structure),
and that image count == pickle count. Exits 1 if any corruption is found.

Usage:  python check_integrity.py <dataset_root> [--episodes epX epY ...]
"""
import os, sys, pickle, argparse
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--episodes", nargs="*", default=None, help="limit to these episode dir names")
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

root = args.root
eps = args.episodes or sorted(d for d in os.listdir(root)
                              if d.startswith("episode_") and os.path.isdir(os.path.join(root, d)))

bad_jpg, bad_pkl, mismatch, empty = [], [], [], []
tot_jpg = tot_pkl = n_empty_frames = 0

for ep in eps:
    im_dir = os.path.join(root, ep, "image")
    pk_dir = os.path.join(root, ep, "pickle_nomad")
    if not (os.path.isdir(im_dir) and os.path.isdir(pk_dir)):
        mismatch.append(f"{ep}: missing image/ or pickle_nomad/")
        continue
    jpgs = [f for f in os.listdir(im_dir) if f.endswith(".jpg")]
    pkls = [f for f in os.listdir(pk_dir) if f.endswith(".pkl")]
    # skip any in-progress files
    if any(f.endswith(".incomplete") for f in os.listdir(im_dir) + os.listdir(pk_dir)):
        print(f"[skip] {ep}: has .incomplete (still downloading)")
        continue
    if len(jpgs) == 0 or len(pkls) == 0:
        empty.append(f"{ep}: jpg={len(jpgs)} pkl={len(pkls)}")
    if len(jpgs) != len(pkls):
        mismatch.append(f"{ep}: jpg={len(jpgs)} != pkl={len(pkls)}")
    # verify jpgs
    for f in jpgs:
        tot_jpg += 1
        p = os.path.join(im_dir, f)
        try:
            with Image.open(p) as im:
                im.verify()            # detects truncation/corruption without full decode
        except Exception as e:
            bad_jpg.append(f"{ep}/{f}: {str(e)[:60]}")
    # verify pkls
    for f in pkls:
        tot_pkl += 1
        p = os.path.join(pk_dir, f)
        try:
            d = pickle.load(open(p, "rb"))
            assert isinstance(d, list), "not a list"
            if len(d) == 0:
                n_empty_frames += 1              # frame with no object annotation -> valid, loader skips it
            else:
                assert isinstance(d[0], dict) and "nomad_traj_norm" in d[0] and "prompt" in d[0], "missing keys"
        except Exception as e:
            bad_pkl.append(f"{ep}/{f}: {str(e)[:60]}")
    if not args.quiet:
        print(f"[ok] {ep}: {len(jpgs)} jpg, {len(pkls)} pkl")

print("\n===== SUMMARY =====")
print(f"episodes checked : {len(eps)}")
print(f"jpg verified     : {tot_jpg}   (bad: {len(bad_jpg)})")
print(f"pkl verified     : {tot_pkl}   (bad: {len(bad_pkl)}, empty-annotation frames: {n_empty_frames})")
print(f"count mismatch   : {len(mismatch)}")
print(f"empty episodes   : {len(empty)}")
for tag, lst in [("BAD JPG", bad_jpg), ("BAD PKL", bad_pkl), ("MISMATCH", mismatch), ("EMPTY", empty)]:
    for x in lst[:20]:
        print(f"  [{tag}] {x}")
    if len(lst) > 20:
        print(f"  [{tag}] ... +{len(lst)-20} more")

n_bad = len(bad_jpg) + len(bad_pkl) + len(mismatch) + len(empty)
if n_bad == 0:
    print("\nRESULT: CLEAN ✅ (no corruption)")
    sys.exit(0)
else:
    print(f"\nRESULT: {n_bad} PROBLEM(S) FOUND ❌")
    sys.exit(1)
