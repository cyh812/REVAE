import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def build_clevr_vocab_and_maxcount(clevr_root: str, split: str = "train"):
    """
    Scan CLEVR scenes and return:
      - colors: sorted list of all colors appearing in this split
      - shapes: sorted list of all shapes appearing in this split
      - max_objects: max number of objects in a single image (in this split)
      - scenes: raw scenes list (for later use)
    """
    scenes_path = os.path.join(clevr_root, "scenes", f"CLEVR_{split}_scenes.json")
    with open(scenes_path, "r", encoding="utf-8") as f:
        scenes_json = json.load(f)

    scenes = scenes_json["scenes"]

    color_set = set()
    shape_set = set()
    max_objects = 0

    for scene in scenes:
        objs = scene["objects"]
        max_objects = max(max_objects, len(objs))
        for obj in objs:
            color_set.add(obj["color"])
            shape_set.add(obj["shape"])

    colors = sorted(color_set)
    shapes = sorted(shape_set)

    return colors, shapes, max_objects, scenes

def build_global_vocab_and_maxcount(clevr_root: str, splits=("train","val","test")):
    all_colors, all_shapes = set(), set()
    global_max = 0
    scenes_by_split = {}

    for sp in splits:
        colors, shapes, max_objects, scenes = build_clevr_vocab_and_maxcount(clevr_root, sp)
        all_colors.update(colors)
        all_shapes.update(shapes)
        global_max = max(global_max, max_objects)
        scenes_by_split[sp] = scenes

    return sorted(all_colors), sorted(all_shapes), global_max, scenes_by_split


class CLEVRMultiLabelByImage(Dataset):
    """
    One sample = one image with 3 labels:
      - color_multi_hot: [num_colors]
      - shape_multi_hot: [num_shapes]
      - count_one_hot:  [max_objects+1]  (index = number of objects)
    """
    def __init__(
        self,
        clevr_root: str,
        split: str,
        colors: List[str],
        shapes: List[str],
        max_objects: int,
        transform=None,
    ):
        assert split in ["train","val","test"]
        self.clevr_root = clevr_root
        self.split = split
        self.colors = colors
        self.shapes = shapes
        self.max_objects = max_objects

        self.color_to_idx = {c:i for i,c in enumerate(colors)}
        self.shape_to_idx = {s:i for i,s in enumerate(shapes)}

        scenes_path = os.path.join(clevr_root, "scenes", f"CLEVR_{split}_scenes.json")
        with open(scenes_path, "r", encoding="utf-8") as f:
            self.scenes = json.load(f)["scenes"]

        self.img_dir = os.path.join(clevr_root, "images", split)

        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]

        # --- image filename ---
        img_fn = scene.get("image_filename", None)
        if img_fn is None:
            # fallback if absent
            img_fn = f"CLEVR_{self.split}_{scene['image_index']:06d}.png"
        img_path = os.path.join(self.img_dir, img_fn)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # --- build labels ---
        color_mh = torch.zeros(len(self.colors), dtype=torch.float32)
        shape_mh = torch.zeros(len(self.shapes), dtype=torch.float32)

        num_objs = len(scene["objects"])
        if num_objs > self.max_objects:
            raise ValueError(f"Found num_objs={num_objs} > max_objects={self.max_objects}. "
                             f"Check your max_objects computation.")

        count_oh = torch.zeros(self.max_objects + 1, dtype=torch.float32)
        count_oh[num_objs] = 1.0

        for obj in scene["objects"]:
            c = obj["color"]
            s = obj["shape"]
            # set multi-hot
            color_mh[self.color_to_idx[c]] = 1.0
            shape_mh[self.shape_to_idx[s]] = 1.0

        return img, color_mh, shape_mh, count_oh, img_fn