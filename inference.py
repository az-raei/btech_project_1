from ultralytics import YOLO
import os

model = YOLO("/home/user/obdet/runs/detect/train11/weights/best.pt") #replace train{number} depending on model

def save_single_class(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for r in results:
        img_name = os.path.basename(r.path).replace(".jpg", ".txt")
        out_path = os.path.join(save_dir, img_name)

        with open(out_path, "w") as f:
            for box in r.boxes:
                cls = 0  # force single class
                x, y, w, h = box.xywhn[0].tolist()
                conf = float(box.conf)
                f.write(f"{cls} {x} {y} {w} {h} {conf}\n")


# NORMAL
res_n = model.predict(
    source="/home/user/obdet/dataset/images/test/normal/",
    conf=0.4,
    verbose=False
)

save_single_class(res_n, "/home/user/obdet/runs/detect/predict_normal_m/labels")


# OCCLUDED
res_o = model.predict(
    source="/home/user/obdet/dataset/images/test/occluded/",
    conf=0.4,
    verbose=False
)

save_single_class(res_o, "/home/user/obdet/runs/detect/predict_occluded_m/labels")

print("YOLOv8m inference done")
