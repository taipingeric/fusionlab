from typing import Sequence
import json
import numpy as np
import os
from glob import glob
from tqdm.auto import tqdm

def convert_labelme_json2mask(
        class_names: Sequence[str], 
        json_dir: str, 
        output_dir: str, 
        single_mask: bool = True
    ):
    """
    Convert labelme json files to mask files(.png)

    Args:
        class_names (list): list of class names, background class must be included at first
        json_dir (str): path to json files directory
        output_dir (str): path to output directory
        single_mask (bool): if True, save single mask file with class index(uint8), otherwise save multiple mask files with class index(uint8)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python package is not installed")

    num_classes = len(class_names)
    if num_classes > 255:
        raise ValueError("Maximum number of classes is 255")
    
    cls_map = {name: i for i, name in enumerate(class_names)}
    print(f"Number of classes: {num_classes}")
    print("class name to index: ", cls_map)
    json_paths = glob(os.path.join(json_dir, '*.json'))

    for path in tqdm(json_paths):
        json_data = json.load(open(path))
        h = json_data['imageHeight']
        w = json_data['imageWidth']

        # Draw Object mask
        mask = np.zeros((len(class_names), h, w), dtype=np.uint8)
        for shape in json_data['shapes']:
            if shape["shape_type"] != "polygon":
                continue
            cls_name = shape['label']
            cls_idx = cls_map[cls_name]
            points = shape['points']
            cv2.fillPoly(
                mask[cls_idx],
                np.array([points], dtype=np.int32),
                255
            )
        # update backgroud mask
        mask[0] = 255-np.max(mask[1:], axis=0)
        # Save Mask File
        filename = ".".join(os.path.split(path)[-1].split('.')[:-1])
        if single_mask:
            mask_single = np.argmax(mask, axis=0).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{filename}.png"), mask_single)
        else:
            for i, m in enumerate(mask):
                cv2.imwrite(os.path.join(output_dir, f'{filename}_{i:03d}.png'), m.astype(np.uint8))

if __name__ == '__main__':
    convert_labelme_json2mask(
        ['bg', 'dog', 'cat'],
        "json",
        "mask",
        single_mask=True,
    )