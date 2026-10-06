#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""葫芦 HSV 阈值颜色识别程序。

程序面向无图形界面的 Ubuntu 虚拟机，不调用 cv2.imshow。
默认读取 ~/exam/opencv 中的两张考核图片，并将结果保存到 outputs。

运行示例：
    cd ~/exam/opencv
    python3 color_threshold.py --output-dir outputs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "缺少 OpenCV 或 NumPy，请先执行："
        "python3 -m pip install -r requirements.txt"
    ) from exc


# 零基础操作手册中的默认图片目录。
# 图片在其他位置时，可以通过 --image-dir 参数指定。
DEFAULT_IMAGE_DIR = Path.home() / "exam" / "opencv"

# OpenCV 的 HSV 取值范围为 H=0~179、S=0~255、V=0~255。
# 黄色范围延伸到 H=43，是为了包含 TZ1_No100(2).jpg 中偏黄绿色的葫芦上端。
PRESETS: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red": [
        ((0, 140, 80), (7, 255, 255)),
        ((173, 140, 80), (179, 255, 255)),
    ],
    "orange": [
        ((4, 150, 80), (24, 255, 255)),
    ],
    "yellow": [
        ((25, 80, 80), (43, 255, 255)),
    ],
    "blue": [
        ((90, 80, 50), (130, 255, 255)),
    ],
}

# 绘制彩色掩膜和检测框时使用的 BGR 颜色。
DRAW_BGR: Dict[str, Tuple[int, int, int]] = {
    "red": (40, 40, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 230, 255),
    "blue": (255, 100, 0),
}

# 每张图片的检测配置。
# ROI 格式为 x, y, width, height，用于排除书籍、木桌、假叶等无关区域。
IMAGE_PROFILES: Dict[str, Dict[str, object]] = {
    "TZ1_No002(1).jpg": {
        "colors": ("red", "orange", "blue"),
        "rois": {
            "red": (1000, 250, 550, 800),
            "orange": (120, 680, 430, 590),
            "blue": None,
        },
        "merge_gap": 41,
    },
    "TZ1_No100(2).jpg": {
        "colors": ("blue", "yellow", "red", "orange"),
        "rois": {
            "blue": (40, 700, 250, 350),
            "yellow": (300, 650, 1200, 450),
            "red": (620, 700, 300, 350),
            "orange": (1000, 750, 250, 350),
        },
        "merge_gap": 41,
    },
}

Roi = Tuple[int, int, int, int]


def parse_roi(text: str) -> Roi:
    """解析 x,y,w,h 格式的 ROI 参数。"""
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI 格式必须为 x,y,w,h") from exc

    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI 格式必须为 x,y,w,h")
    if values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("ROI 的宽度和高度必须大于 0")
    return values


def parse_colors(text: str) -> Tuple[str, ...]:
    """解析以英文逗号分隔的颜色名称。"""
    colors = tuple(value.strip().lower() for value in text.split(",") if value.strip())
    invalid = [color for color in colors if color not in PRESETS]
    if invalid:
        raise argparse.ArgumentTypeError(
            "存在未知颜色：{0}；可选值为：{1}".format(
                ", ".join(invalid), ", ".join(sorted(PRESETS))
            )
        )
    if not colors:
        raise argparse.ArgumentTypeError("至少需要指定一种颜色")
    return colors


def imread_color(path: Path) -> Optional[np.ndarray]:
    """读取彩色图片，并兼容包含中文或空格的路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def build_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """合并一种颜色的所有 HSV 范围，生成二值掩膜。"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in PRESETS[color]:
        lower_array = np.array(lower, dtype=np.uint8)
        upper_array = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_array, upper_array))
    return mask


def remove_noise(mask: np.ndarray) -> np.ndarray:
    """通过形态学开闭运算去除噪点并填补小空洞。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def apply_roi(mask: np.ndarray, roi: Optional[Roi]) -> np.ndarray:
    """只保留指定 ROI 内的掩膜像素。"""
    if roi is None:
        return mask

    x, y, width, height = roi
    image_height, image_width = mask.shape[:2]
    x1 = max(0, min(x, image_width - 1))
    y1 = max(0, min(y, image_height - 1))
    x2 = max(x1 + 1, min(x + width, image_width))
    y2 = max(y1 + 1, min(y + height, image_height))

    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = 255
    return cv2.bitwise_and(mask, roi_mask)


def merge_regions(mask: np.ndarray, gap: int) -> np.ndarray:
    """连接被葫芦腰部或反光高光分开的同一目标区域。"""
    if gap <= 1:
        return mask
    size = gap if gap % 2 == 1 else gap + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def find_boxes(mask: np.ndarray, min_area: float) -> List[Tuple[float, Roi]]:
    """查找外部轮廓，并返回轮廓面积和边界框。"""
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    boxes: List[Tuple[float, Roi]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        boxes.append((area, (x, y, width, height)))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return boxes


def save_image(path: Path, image: np.ndarray) -> None:
    """保存图片，并兼容包含中文的路径。"""
    suffix = path.suffix if path.suffix else ".png"
    try:
        success, encoded = cv2.imencode(suffix, image)
    except cv2.error as exc:
        raise RuntimeError("图片编码失败：{0}".format(path)) from exc
    if not success:
        raise RuntimeError("图片编码失败：{0}".format(path))
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise RuntimeError("图片保存失败：{0}".format(path)) from exc


def process_image(
    image_path: Path,
    output_dir: Path,
    colors: Sequence[str],
    rois: Dict[str, Optional[Roi]],
    min_area: float,
    merge_gap: int,
    padding: int,
) -> Dict[str, object]:
    """处理单张图片，并保存掩膜、可视化结果和 JSON 数据。"""
    image = imread_color(image_path)
    if image is None:
        raise RuntimeError("无法读取图片：{0}".format(image_path))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    color_layer = np.zeros_like(image)
    detections: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}
    individual_masks: List[Tuple[str, np.ndarray]] = []

    for color in colors:
        # 按当前颜色生成阈值掩膜。
        mask = build_mask(hsv, color)
        mask = remove_noise(mask)
        mask = apply_roi(mask, rois.get(color))
        mask = merge_regions(mask, merge_gap)

        # 查找有效轮廓并记录检测结果。
        boxes = find_boxes(mask, min_area)
        counts[color] = len(boxes)
        individual_masks.append((color, mask))
        combined_mask[mask > 0] = 255
        color_layer[mask > 0] = DRAW_BGR[color]

        for area, box in boxes:
            x, y, width, height = box
            detections.append(
                {
                    "color": color,
                    "area": round(area, 2),
                    "box": [int(x), int(y), int(width), int(height)],
                    "roi": list(rois[color]) if rois.get(color) is not None else None,
                }
            )

    # 将颜色掩膜叠加到原图上，便于观察识别效果。
    active = combined_mask > 0
    overlay = image.copy()
    if np.any(active):
        tinted = cv2.addWeighted(image, 0.58, color_layer, 0.42, 0)
        overlay[active] = tinted[active]

    # 绘制边界框和英文颜色标签。
    # OpenCV 的 putText 不直接支持中文，因此标签保留 red、orange 等英文名称。
    image_height, image_width = image.shape[:2]
    for detection in detections:
        color = str(detection["color"])
        x, y, width, height = detection["box"]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image_width - 1, x + width + padding)
        y2 = min(image_height - 1, y + height + padding)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), DRAW_BGR[color], 3)

        label = color
        text_y = y1 - 10 if y1 >= 32 else y1 + 28
        cv2.putText(
            overlay,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            DRAW_BGR[color],
            2,
            cv2.LINE_AA,
        )

    # 保存该图片对应的所有输出文件。
    stem = image_path.stem
    save_image(output_dir / (stem + "_result.png"), overlay)
    save_image(output_dir / (stem + "_mask.png"), combined_mask)
    save_image(output_dir / (stem + "_color_mask.png"), color_layer)
    for color, mask in individual_masks:
        save_image(output_dir / (stem + "_" + color + "_mask.png"), mask)

    metadata: Dict[str, object] = {
        "image": image_path.name,
        "image_size": [int(image_width), int(image_height)],
        "parameters": {
            "min_area": min_area,
            "merge_gap": merge_gap,
            "padding": padding,
            "hsv_presets": PRESETS,
        },
        "counts": counts,
        "total_targets": sum(counts.values()),
        "detections": detections,
    }

    json_path = output_dir / (stem + "_result.json")
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "[完成] {0}：共检测到 {1} 个目标 {2}".format(
            image_path.name, metadata["total_targets"], counts
        )
    )
    print("       检测效果图：{0}".format(output_dir / (stem + "_result.png")))
    print("       彩色掩膜：{0}".format(output_dir / (stem + "_color_mask.png")))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对两张考核图片进行无图形界面的 HSV 颜色阈值识别。"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="包含两张 JPG 图片的目录，默认为 ~/exam/opencv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="保存掩膜、效果图和 JSON 数据的目录",
    )
    parser.add_argument(
        "--image",
        action="append",
        choices=sorted(IMAGE_PROFILES),
        dest="images",
        help="只处理指定图片，可以重复使用该参数",
    )
    parser.add_argument(
        "--colors",
        type=parse_colors,
        help="临时指定颜色，例如 red,orange,blue,yellow",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        help="临时对所有颜色使用统一 ROI，格式为 x,y,w,h",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1000.0,
        help="最小轮廓面积，默认值为 1000",
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=None,
        help="合并同一葫芦断裂区域时使用的像素间距，默认读取图片配置",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="检测框向外扩展的像素数，默认值为 5",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_dir = args.image_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise SystemExit(
            "图片目录不存在：{0}\n"
            "请使用 --image-dir 指定虚拟机中的图片目录。".format(image_dir)
        )

    selected_images = args.images if args.images else list(IMAGE_PROFILES)
    missing = [name for name in selected_images if not (image_dir / name).is_file()]
    if missing:
        raise SystemExit("缺少输入图片：{0}".format(", ".join(missing)))

    all_metadata: List[Dict[str, object]] = []
    for image_name in selected_images:
        profile = IMAGE_PROFILES[image_name]
        colors = args.colors if args.colors else tuple(profile["colors"])

        profile_rois = dict(profile.get("rois", {}))
        if args.roi is not None:
            rois = {color: args.roi for color in colors}
        else:
            rois = {color: profile_rois.get(color) for color in colors}

        merge_gap = (
            args.merge_gap
            if args.merge_gap is not None
            else int(profile.get("merge_gap", 41))
        )
        all_metadata.append(
            process_image(
                image_dir / image_name,
                output_dir,
                colors,
                rois,
                args.min_area,
                merge_gap,
                args.padding,
            )
        )

    # 汇总两张图片的检测数量、坐标和参数。
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "image_dir": str(image_dir),
                "output_dir": str(output_dir),
                "results": all_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[完成] 汇总结果：{0}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())