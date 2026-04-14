import os
import xml.etree.ElementTree as ET

base_dir = "/home/user/obdet/dataset"

for split in ["train", "val"]:
    xml_dir = os.path.join(base_dir, "labels", split)

    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        txt_path = xml_path.replace(".xml", ".txt")

        with open(txt_path, "w") as f:
            for obj in root.findall("object"):

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # convert to YOLO format
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h

                # single class = 0
                f.write(f"0 {x_center} {y_center} {bw} {bh}\n")



