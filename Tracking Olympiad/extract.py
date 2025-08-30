import os
import glob
import cv2
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────
VIDEO_PATTERN = '/home/hpc/ptfs/ptfs272h/Traco/training/training*.mp4' 
OUT_IMG_DIR   = '/home/hpc/ptfs/ptfs272h/Traco/images'
# ───────────────────────────────────────────────────────────────────

# prepare output dir
os.makedirs(OUT_IMG_DIR, exist_ok=True)

# find all video files
video_paths = sorted(glob.glob(VIDEO_PATTERN))

for vid_path in video_paths:
    basename = os.path.splitext(os.path.basename(vid_path))[0]
    cap = cv2.VideoCapture(vid_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # extract each frame
    for t in tqdm(range(nframes), desc=f"Extracting {basename}"):
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"{basename}_frame{t:04d}.jpg"
        out_path = os.path.join(OUT_IMG_DIR, fname)
        cv2.imwrite(out_path, frame)

    cap.release()

print(f"Done! Extracted frames to {OUT_IMG_DIR}")
