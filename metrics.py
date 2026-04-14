import os

gt_normal = "/home/user/obdet/dataset/labels/test/normal"
gt_occluded = "/home/user/obdet/dataset/labels/test/occluded"

pred_normal = "/home/user/obdet/runs/detect/predict_normal_m/labels"
pred_occluded = "/home/user/obdet/runs/detect/predict_occluded_m/labels"


def load_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            boxes.append(parts[1:5])
    return boxes


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # convert to corners
    def to_xyxy(b):
        x, y, w, h = b
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    b1 = to_xyxy(box1)
    b2 = to_xyxy(box2)

    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate(gt_dir, pred_dir):
    TP = FP = FN = 0

    for file in os.listdir(gt_dir):
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        gt_boxes = load_boxes(gt_path)
        pred_boxes = load_boxes(pred_path)

        matched = set()

        for p in pred_boxes:
            found = False
            for i, g in enumerate(gt_boxes):
                if i in matched:
                    continue
                if iou(p, g) > 0.5:
                    TP += 1
                    matched.add(i)
                    found = True
                    break
            if not found:
                FP += 1

        FN += len(gt_boxes) - len(matched)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    return precision, recall, TP, FP, FN


print("\nFINAL RESULTS\n")

p, r, tp, fp, fn = evaluate(gt_normal, pred_normal)
print("NORMAL:")
print(f"  Precision: {p:.3f}")
print(f"  Recall:    {r:.3f}")
print(f"  TP={tp}, FP={fp}, FN={fn}\n")

p, r, tp, fp, fn = evaluate(gt_occluded, pred_occluded)
print("OCCLUDED:")
print(f"  Precision: {p:.3f}")
print(f"  Recall:    {r:.3f}")
print(f"  TP={tp}, FP={fp}, FN={fn}")
