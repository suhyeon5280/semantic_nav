"""
Language augmentation for direction + OOD-robust instruction fine-tuning.

Two ingredients:
 1) warp_trajectory: bend a REAL (8,4) trajectory left/right (or keep straight) so the
    direction COMMAND controls the target — supervision that real single-trajectory data
    lacks (see the capacity probe). Bending a real path keeps it more scene-consistent
    than a pure synthetic arc.
 2) Diverse phrasing banks per direction concept, split into TRAIN and OOD (held-out)
    phrasings. Training on many surface forms of the same concept makes the policy respond
    to the CLIP *semantic region* rather than exact strings, so it generalizes to unseen
    phrasings. Evaluate with the OOD bank to measure phrasing-level robustness.

Scope (honest): this targets DIRECTION concepts and PHRASING-level OOD (unseen wordings of
known concepts). Concept-level OOD (entirely new command types) still needs richer sources
(VLM/LLM data, distillation).
"""
import torch

# direction -> lateral sign for the warp (+ = left in the (x_fwd, y_left, cos, sin) convention)
_DIR_SIGN = {"left": +1.0, "right": -1.0, "straight": 0.0}

# phrasing banks: TRAIN forms are seen during training; OOD forms are HELD OUT for eval.
# TRAIN is intentionally large & varied (LLM-generated "language teacher") so the policy keys on
# the CLIP *semantic region* of a direction, not exact strings -> phrasing-level OOD robustness.
PHRASINGS_TRAIN = {
    "left": ["go left", "turn left", "veer left", "head to the left", "move to the left",
             "steer left", "take a left", "go to the left side", "curve left", "angle left",
             "shift to the left", "drift left", "bank left", "go leftward", "make a left"],
    "right": ["go right", "turn right", "veer right", "head to the right", "move to the right",
              "steer right", "take a right", "go to the right side", "curve right", "angle right",
              "shift to the right", "drift right", "bank right", "go rightward", "make a right"],
    "straight": ["go straight", "keep going straight", "continue forward", "head straight ahead",
                 "move forward", "go straight ahead", "keep moving forward", "stay straight",
                 "drive forward", "proceed forward", "keep straight", "advance forward"],
}
PHRASINGS_OOD = {  # held out: different register / more distant wordings, never seen in training
    "left": ["bear left", "keep to the left", "swing left", "hang a left", "hug the left",
             "favor the left side", "ease to the left", "peel off left"],
    "right": ["bear right", "keep to the right", "swing right", "hang a right", "hug the right",
              "favor the right side", "ease to the right", "peel off right"],
    "straight": ["stay the course", "carry on ahead", "hold your line", "press on forward", "keep the heading"],
}
DIRECTIONS = list(_DIR_SIGN.keys())


def sample_phrasing(direction, rng, ood=False):
    bank = PHRASINGS_OOD if ood else PHRASINGS_TRAIN
    return rng.choice(bank[direction])


def warp_trajectory(traj, direction, bend, mode="add"):
    """
    traj: (T,4) tensor = (x_fwd, y_left, cos, sin), normalized units.
    Builds a command-driven lateral bend growing quadratically over the horizon, then
    recomputes heading (cos,sin) from the resulting positions. `bend` = endpoint lateral offset.
      mode="add"     : warp the REAL lateral (real + command bend) — more scene-consistent, but the
                       command signal is a minority of the target (weak for teaching direction).
      mode="replace" : lateral is PURELY command-driven (forward kept from real) — command DOMINATES
                       the target, so it teaches command->direction strongly (probe-style).
    """
    T = traj.shape[0]
    dev = traj.device
    t = torch.linspace(1.0 / T, 1.0, T, device=dev)
    bend_profile = _DIR_SIGN[direction] * bend * (t ** 2)    # 0 at start -> +/-bend at end
    out = traj.clone()
    if mode == "replace":
        out[:, 1] = bend_profile                             # lateral = command only (forward stays real)
    else:
        out[:, 1] = out[:, 1] + bend_profile                 # lateral = real + command bend
    # recompute heading from consecutive (x,y) positions
    x, y = out[:, 0], out[:, 1]
    x0 = torch.cat([torch.zeros(1, device=dev), x[:-1]])
    y0 = torch.cat([torch.zeros(1, device=dev), y[:-1]])
    ang = torch.atan2(y - y0, (x - x0) + 1e-6)
    out[:, 2] = torch.cos(ang)
    out[:, 3] = torch.sin(ang)
    return out
