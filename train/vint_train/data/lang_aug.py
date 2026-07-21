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

# phrasing banks: TRAIN forms are seen during training; OOD forms are held out for eval
PHRASINGS_TRAIN = {
    "left":     ["go left", "turn left", "veer left", "head to the left", "move to the left", "steer left"],
    "right":    ["go right", "turn right", "veer right", "head to the right", "move to the right", "steer right"],
    "straight": ["go straight", "keep going straight", "continue forward", "head straight ahead", "move forward"],
}
PHRASINGS_OOD = {
    "left":     ["bear left", "keep to the left", "swing left", "hang a left"],
    "right":    ["bear right", "keep to the right", "swing right", "hang a right"],
    "straight": ["stay the course", "carry on ahead", "proceed straight"],
}
DIRECTIONS = list(_DIR_SIGN.keys())


def sample_phrasing(direction, rng, ood=False):
    bank = PHRASINGS_OOD if ood else PHRASINGS_TRAIN
    return rng.choice(bank[direction])


def warp_trajectory(traj, direction, bend):
    """
    traj: (T,4) tensor = (x_fwd, y_left, cos, sin), normalized units.
    Adds a command-driven lateral bend growing quadratically over the horizon, then
    recomputes heading (cos,sin) from the warped positions. `bend` = endpoint lateral offset.
    """
    T = traj.shape[0]
    dev = traj.device
    t = torch.linspace(1.0 / T, 1.0, T, device=dev)
    offset = _DIR_SIGN[direction] * bend * (t ** 2)          # 0 at start -> +/-bend at end
    out = traj.clone()
    out[:, 1] = out[:, 1] + offset                           # warp the lateral channel
    # recompute heading from consecutive (x,y) positions
    x, y = out[:, 0], out[:, 1]
    x0 = torch.cat([torch.zeros(1, device=dev), x[:-1]])
    y0 = torch.cat([torch.zeros(1, device=dev), y[:-1]])
    ang = torch.atan2(y - y0, (x - x0) + 1e-6)
    out[:, 2] = torch.cos(ang)
    out[:, 3] = torch.sin(ang)
    return out
