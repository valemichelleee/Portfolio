import csv
from ultralytics import YOLO

# 1. Load your model
model = YOLO("runs/train/hexbug_v8s_tuned/weights/best.pt")

# 2. Open CSV and write header
with open('detections.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['idx', 'frame_id', 'x', 'y', 'conf'])

    idx = 0
    # 3. Run inference in stream mode so you get one Results object per frame
    for frame_id, result in enumerate(model.predict(source='arXiv2020-RIFE/test/test001.mp4',
                                                    device=0,
                                                    conf=0.25,
                                                    stream=True)):
        # result.boxes is a list of Boxes for this frame
        for box in result.boxes:
            # box.xywh gives [x_center, y_center, width, height]
            x_center, y_center, _, _ = box.xywh[0].tolist()
            conf      = float(box.conf[0])  # confidence score

            # 4. Write a row per detection
            writer.writerow([idx, frame_id, x_center, y_center, conf])
            idx += 1
