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
    One sample = one image with labels:
      - color_multi_hot: [num_colors]
      - shape_multi_hot: [num_shapes]
      - count_one_hot:  [max_objects+1]  (index = number of objects)
      - color_count:    [num_colors, max_objects+1] (each row one-hot count 0..max)
      - shape_count:    [num_shapes, max_objects+1]
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

        # --- image filename/path ---
        img_fn = scene.get("image_filename", None)
        if img_fn is None:
            img_fn = f"CLEVR_{self.split}_{scene['image_index']:06d}.png"
        img_path = os.path.join(self.img_dir, img_fn)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # --- basic labels ---
        num_colors = len(self.colors)
        num_shapes = len(self.shapes)

        color_mh = torch.zeros(num_colors, dtype=torch.float32)
        shape_mh = torch.zeros(num_shapes, dtype=torch.float32)

        num_objs = len(scene["objects"])
        if num_objs > self.max_objects:
            raise ValueError(f"Found num_objs={num_objs} > max_objects={self.max_objects}. "
                             f"Check your max_objects computation.")

        count_oh = torch.zeros(self.max_objects + 1, dtype=torch.float32)
        count_oh[num_objs] = 1.0

        # --- NEW: per-color / per-shape counts (integer counters first) ---
        color_cnt_int = torch.zeros(num_colors, dtype=torch.long)  # 每个颜色出现次数
        shape_cnt_int = torch.zeros(num_shapes, dtype=torch.long)  # 每个形状出现次数

        for obj in scene["objects"]:
            c = obj["color"]
            s = obj["shape"]

            ci = self.color_to_idx[c]
            si = self.shape_to_idx[s]

            color_mh[ci] = 1.0
            shape_mh[si] = 1.0

            color_cnt_int[ci] += 1
            shape_cnt_int[si] += 1

        # --- convert integer counts to one-hot rows (num_x, max+1) ---
        color_count = torch.zeros(num_colors, self.max_objects + 1, dtype=torch.float32)
        shape_count = torch.zeros(num_shapes, self.max_objects + 1, dtype=torch.float32)

        # 每行 one-hot：index = count
        color_count[torch.arange(num_colors), color_cnt_int.clamp_(0, self.max_objects)] = 1.0
        shape_count[torch.arange(num_shapes), shape_cnt_int.clamp_(0, self.max_objects)] = 1.0

        # ✅ 按你的要求：去掉 img_fn，新增两个绑定标签
        return img, color_mh, shape_mh, count_oh, color_count, shape_count
