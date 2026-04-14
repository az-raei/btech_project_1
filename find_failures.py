import os
import cv2

gt_dir = "/home/user/obdet/dataset/labels/test/occluded"
pred_dir = "/home/user/obdet/runs/detect/predict_occluded_m/labels"
img_dir = "/home/user/obdet/dataset/images/test/occluded"

output_dir = "/home/user/obdet/failures"
os.makedirs(output_dir, exist_ok=True)


def load_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            boxes.append(parts[1:5])
    return boxes


def to_xyxy(box, w, h):
    x, y, bw, bh = box
    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)
    return [x1, y1, x2, y2]


def iou(b1, b2):
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])

    inter = max(0, xi2-xi1) * max(0, yi2-yi1)

    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])

    union = a1 + a2 - inter
    return inter/union if union > 0 else 0


for file in os.listdir(gt_dir):
    img_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    gt_boxes = [to_xyxy(b, w, h) for b in load_boxes(os.path.join(gt_dir, file))]
    pred_boxes = [to_xyxy(b, w, h) for b in load_boxes(os.path.join(pred_dir, file))]

    matched = set()

    # match
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            if j in matched:
                continue
            if iou(p, g) > 0.5:
                matched.add(j)
                cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0,255,0), 2)  # green
                break
        else:
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255,0,0), 2)  # blue

    for j, g in enumerate(gt_boxes):
        if j not in matched:
            cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (0,0,255), 2)  # red

    cv2.imwrite(os.path.join(output_dir, file.replace(".txt", ".jpg")), img)

print("Failure images saved")
