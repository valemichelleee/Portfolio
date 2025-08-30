import os, glob, json, cv2
from pathlib import Path
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────
IMG_DIR   = '/home/hpc/ptfs/ptfs272h/Traco/images'
CSV_DIR   = '/home/hpc/ptfs/ptfs272h/Traco/data'
LBL_DIR   = '/home/hpc/ptfs/ptfs272h/Traco/labels'
BOX_SIZE  = 64  # same as before
# ────────────────────────────────────────────────────────────────────

os.makedirs(LBL_DIR, exist_ok=True)

# For each video’s CSV:
for csv_path in sorted(glob.glob(f"{CSV_DIR}/training*.csv")):
    df = pd.read_csv(csv_path)  # cols: t, hexbug, x, y
    stem = Path(csv_path).stem   # e.g. "training01"

    # Group coords by frame index
    by_frame = df.groupby('t')[['x','y']].apply(lambda g: g.values.tolist()).to_dict()

    # For each frame that actually exists as a JPEG:
    for img_path in sorted(glob.glob(f"{IMG_DIR}/{stem}_frame*.jpg")):
        fname = Path(img_path).stem          # e.g. "training01_frame0000"
        t = int(fname.split('_frame')[-1])   # extract frame index

        # Load to get dims
        im = cv2.imread(img_path)
        H, W = im.shape[:2]

        lines = []
        for x,y in by_frame.get(t, []):
            x0 = max(0, min(x - BOX_SIZE/2, W - BOX_SIZE))
            y0 = max(0, min(y - BOX_SIZE/2, H - BOX_SIZE))
            xc = (x0 + BOX_SIZE/2) / W
            yc = (y0 + BOX_SIZE/2) / H
            wn = BOX_SIZE / W
            hn = BOX_SIZE / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        # Write label file
        with open(os.path.join(LBL_DIR, f"{fname}.txt"), 'w') as f:
            f.write("\n".join(lines))
