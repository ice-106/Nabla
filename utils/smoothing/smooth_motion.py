# ── Full pipeline smoothing v3: fixed hands + locked camera ─────────────
import glob, os, math, json
import numpy as np

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = np.vectorize(self._alpha)(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

def smooth_oneeuro(sequence, freq=30, min_cutoff=0.3, beta=0.01):
    orig_shape = sequence.shape
    T = orig_shape[0]
    flat = sequence.reshape(T, -1)
    filt = OneEuroFilter(freq, min_cutoff, beta)
    fwd = np.empty_like(flat)
    for t in range(T):
        fwd[t] = filt(flat[t])
    filt2 = OneEuroFilter(freq, min_cutoff, beta)
    out = np.empty_like(flat)
    for t in range(T - 1, -1, -1):
        out[t] = filt2(fwd[t])
    return out.reshape(orig_shape)

# min_cutoff = Hz. Higher = more motion allowed through.
_PARAM_SETTINGS = {
    'transl':           (0.05, 0.005),   # very smooth translation
    'global_orient':    (0.15, 0.01),    # smooth root rotation
    'body_pose':        (0.40, 0.02),    # body can move
    'left_hand_pose':   (0.15, 0.01),    # hands move freely
    'right_hand_pose':  (0.15, 0.01),    # hands move freely
    'jaw_pose':         (0.50, 0.01),
    'leye_pose':        (0.50, 0.01),
    'reye_pose':        (0.50, 0.01),
    'expression':       (0.50, 0.01),
}
_ARM_START = 12 * 3
_WRIST_START = 19 * 3  # = 57
_NPZ_PARAM_KEYS = list(_PARAM_SETTINGS.keys()) + ['betas']

def smooth_all(base_dir, output_suffix="smoothed", freq=30):
    smplx_dir = os.path.join(base_dir, "smplx")
    meta_dir  = os.path.join(base_dir, "meta")
    smplx_out = os.path.join(base_dir, f"smplx_{output_suffix}")
    meta_out  = os.path.join(base_dir, f"meta_{output_suffix}")

    # ── 1. Smooth SMPL-X params ─────────────────────────────────────────
    all_npz = sorted(glob.glob(os.path.join(smplx_dir, '*.npz')))
    if not all_npz:
        raise FileNotFoundError(f'No .npz in {smplx_dir}')

    groups = {}
    for path in all_npz:
        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.rsplit('_', 1)
        suffix = parts[1] if len(parts) == 2 and parts[1].isdigit() else ''
        groups.setdefault(suffix, []).append(path)

    os.makedirs(smplx_out, exist_ok=True)

    for suffix, paths in groups.items():
        paths = sorted(paths)
        T = len(paths)
        print(f'[npz] person={suffix or "0"}  frames={T}')

        all_data = [dict(np.load(p)) for p in paths]
        available_keys = [k for k in _NPZ_PARAM_KEYS if k in all_data[0]]

        smoothed = {}
        for key in available_keys:
            stacked = np.stack([d[key].reshape(-1) for d in all_data])
            if key == 'betas':
                median = np.median(stacked, axis=0, keepdims=True)
                smoothed[key] = np.repeat(median, T, axis=0)
                print(f'  {key:20s} -> locked to median')
            elif _PARAM_SETTINGS.get(key) is None:
                smoothed[key] = stacked
                print(f'  {key:20s} -> passthrough')
            elif key == 'body_pose':
                mc, b = _PARAM_SETTINGS[key]
                torso_smooth = smooth_oneeuro(stacked[:, :_ARM_START], freq, mc, b)
                arms_raw     = stacked[:, _ARM_START:]
                smoothed[key] = np.concatenate([torso_smooth, arms_raw], axis=1)
                print(f'  {key:20s} -> torso smooth, arms+wrists passthrough')
            else:
                mc, b = _PARAM_SETTINGS[key]
                smoothed[key] = smooth_oneeuro(stacked, freq, mc, b)
                print(f'  {key:20s} -> min_cutoff={mc}  beta={b}')

        for i, p in enumerate(paths):
            out_data = dict(all_data[i])
            for key in available_keys:
                out_data[key] = smoothed[key][i].reshape(all_data[i][key].shape)
            np.savez(os.path.join(smplx_out, os.path.basename(p)), **out_data)

    print(f'[npz] Done -> {smplx_out}')

    # ── 2. Lock meta .json (focal, princpt, bbox → median) ──────────────
    all_json = sorted(glob.glob(os.path.join(meta_dir, '*.json')))
    if not all_json:
        print('[meta] No .json found, skipping')
        return

    T = len(all_json)
    print(f'[meta] frames={T}')

    all_meta = []
    for p in all_json:
        with open(p) as f:
            all_meta.append(json.load(f))

    focals   = np.array([m['focal']   for m in all_meta])    # (T, 2)
    princpts = np.array([m['princpt'] for m in all_meta])    # (T, 2)
    bboxes   = np.array([m['bbox']    for m in all_meta])    # (T, 4)

    # Camera intrinsics are CONSTANT for a fixed camera — use median
    focal_locked   = np.median(focals,   axis=0)
    princpt_locked = np.median(princpts, axis=0)
    # Bbox: smooth aggressively instead of locking (person can move in frame)
    bbox_smooth    = smooth_oneeuro(bboxes, freq, min_cutoff=0.02, beta=0.002)

    print(f'  focal  locked to: [{focal_locked[0]:.0f}, {focal_locked[1]:.0f}]'
          f'  (was {focals[:,0].min():.0f}-{focals[:,0].max():.0f})')
    print(f'  princpt locked to: [{princpt_locked[0]:.1f}, {princpt_locked[1]:.1f}]')
    print(f'  bbox_w range: {bboxes[:,2].min():.0f}-{bboxes[:,2].max():.0f}'
          f' -> {bbox_smooth[:,2].min():.0f}-{bbox_smooth[:,2].max():.0f}')

    os.makedirs(meta_out, exist_ok=True)

    for i, p in enumerate(all_json):
        m = dict(all_meta[i])
        m['focal']   = focal_locked.tolist()
        m['princpt'] = princpt_locked.tolist()
        m['bbox']    = bbox_smooth[i].tolist()
        out_path = os.path.join(meta_out, os.path.basename(p))
        with open(out_path, 'w') as f:
            json.dump(m, f, indent=2)

    print(f'[meta] Done -> {meta_out}')

# ── Run ──────────────────────────────────────────────────────────────────
BASE = "/kaggle/working/SMPLer-X/demo/results/test_video"
smooth_all(BASE)
