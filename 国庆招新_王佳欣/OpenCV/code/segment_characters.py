#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用专用算法分割奶龙和蜡笔小新。

本程序参考了 02_segment_nailong.py 和 03_segment_xiaoxin.py：
- R-C.jpg：HSV/BGR 联合颜色阈值 + 主体多边形 + 形态学处理；
- v2 图片：完整身体多边形 + 红衣服/肤色/头发颜色种子 + 连通区域筛选。

程序默认无图形界面，原手册中的 --rect 和 --no-display 参数仍然兼容。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "缺少 OpenCV 或 NumPy，请先执行："
        "python3 -m pip install -r requirements.txt"
    ) from exc


DEFAULT_IMAGE_DIR = Path.home() / "exam" / "opencv"
DEFAULT_OUTPUT_DIR = DEFAULT_IMAGE_DIR / "outputs"

Rect = Tuple[int, int, int, int]

IMAGE_NAMES = {
    "nailong": "R-C.jpg",
    "xiaoxin": "v2-e08a9d9a3b5831815739d8b65d6c4d76_r.png",
}

# 特殊图片的初始矩形仍保留，主要用于旧命令兼容。
DEFAULT_RECTS = {
    IMAGE_NAMES["nailong"]: (600, 150, 430, 570),
    IMAGE_NAMES["xiaoxin"]: (0, 0, 1050, 1080),
}


def parse_rect(text: str) -> Rect:
    """解析 x,y,w,h 格式的矩形参数。"""
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("矩形格式必须为 x,y,w,h") from exc

    if len(values) != 4:
        raise argparse.ArgumentTypeError("矩形格式必须为 x,y,w,h")
    if values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("矩形宽度和高度必须大于 0")
    return values


def imread_color(path: Path) -> Optional[np.ndarray]:
    """读取彩色图片，并兼容包含中文的路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_image(path: Path, image: np.ndarray) -> None:
    """保存图片，并兼容包含中文的路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() if path.suffix else ".png"
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


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留面积最大的八邻域连通区域。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask)
    largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_id, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """填充目标内部的黑色孔洞。"""
    flood = mask.copy()
    height, width = mask.shape[:2]
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def polish_mask(mask: np.ndarray, open_size: int = 5) -> np.ndarray:
    """通过闭运算和开运算清理掩膜边缘。"""
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, close_kernel, iterations=2
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask


def limit_to_roi(mask: np.ndarray, rect: Rect) -> np.ndarray:
    """只保留指定 ROI 内的掩膜。"""
    x, y, width, height = rect
    result = np.zeros_like(mask)
    result[y : y + height, x : x + width] = mask[
        y : y + height, x : x + width
    ]
    return result


# ============================ 奶龙分割参数 ============================

NAILONG_SIZE = (1280, 720)
NAILONG_ROI = (665, 190, 370, 530)
NAILONG_HSV_LOWER = np.array([10, 60, 130], dtype=np.uint8)
NAILONG_HSV_UPPER = np.array([38, 255, 255], dtype=np.uint8)
NAILONG_RELAXED_LOWER = np.array([5, 35, 100], dtype=np.uint8)
NAILONG_RELAXED_UPPER = np.array([40, 255, 255], dtype=np.uint8)

NAILONG_BODY_POLYGON = np.array(
    [
        [870, 245], [925, 245], [965, 275], [985, 320],
        [985, 380], [970, 425], [1000, 445], [1030, 465],
        [1045, 535], [1045, 575], [1020, 625], [985, 655],
        [975, 720], [760, 720], [740, 655], [722, 580],
        [710, 515], [720, 455], [745, 435], [755, 390],
        [775, 335], [815, 285],
    ],
    dtype=np.int32,
)

NAILONG_ARM_POLYGON = np.array(
    [
        [925, 415], [975, 425], [1020, 450], [1045, 515],
        [1040, 590], [1020, 650], [990, 720], [950, 720],
        [930, 620], [920, 540],
    ],
    dtype=np.int32,
)


def segment_nailong(image: np.ndarray) -> np.ndarray:
    """使用颜色阈值和主体多边形分割奶龙。"""
    actual_size = (image.shape[1], image.shape[0])
    if actual_size != NAILONG_SIZE:
        raise ValueError(
            "R-C.jpg 应为 1280x720，实际为 {0}x{1}".format(*actual_size)
        )

    body_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(body_mask, [NAILONG_BODY_POLYGON], 255)

    arm_mask = np.zeros_like(body_mask)
    cv2.fillPoly(arm_mask, [NAILONG_ARM_POLYGON], 255)
    cv2.ellipse(arm_mask, (995, 580), (45, 140), 0, 0, 360, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    strict_mask = cv2.inRange(
        hsv, NAILONG_HSV_LOWER, NAILONG_HSV_UPPER
    )
    relaxed_mask = cv2.inRange(
        hsv, NAILONG_RELAXED_LOWER, NAILONG_RELAXED_UPPER
    )

    blue, green, red = cv2.split(image)
    red16 = red.astype(np.int16)
    green16 = green.astype(np.int16)
    blue16 = blue.astype(np.int16)
    ratio = green16.astype(np.float32) / np.maximum(
        red16.astype(np.float32), 1.0
    )

    strict_channels = (
        (red16 > 180) & (green16 > 140) & (blue16 < 200)
    )
    strict_channels &= (red16 - blue16) > 45
    strict_channels &= ratio > 0.58
    strict_mask = cv2.bitwise_and(
        strict_mask, strict_channels.astype(np.uint8) * 255
    )

    relaxed_channels = (
        (red16 > 145) & (green16 > 95) & (blue16 < 200)
    )
    relaxed_channels &= (red16 - blue16) > 30
    relaxed_channels &= ratio > 0.50
    relaxed_mask = cv2.bitwise_and(
        relaxed_mask, relaxed_channels.astype(np.uint8) * 255
    )

    mask = cv2.bitwise_or(strict_mask, relaxed_mask)
    mask = cv2.bitwise_and(mask, body_mask)
    mask = limit_to_roi(mask, NAILONG_ROI)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.bitwise_or(mask, arm_mask)
    mask = cv2.bitwise_and(mask, body_mask)

    stem_orange = (
        (red16 > 180) & (green16 < 160) & (blue16 < 90)
    )
    stem_orange |= ratio < 0.82
    rows = np.indices(mask.shape)[0]
    stem_orange &= rows < 310
    mask[stem_orange] = 0
    mask[150:320, 870:895] = 0

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    mask = fill_holes(mask)
    mask = cv2.bitwise_and(mask, body_mask)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(mask) < 1000:
        raise RuntimeError("奶龙掩膜面积过小，请检查图片和 HSV 阈值。")
    return mask

# ============================ 蜡笔小新参数 ============================

XIAOXIN_SIZE = (1920, 1080)

XIAOXIN_FULL_BODY_POLYGON = np.array(
    [
        [250, 320], [285, 235], [390, 185], [520, 165], [650, 180],
        [730, 220], [800, 280], [880, 320], [960, 370], [1030, 450],
        [1070, 550], [1090, 650], [1120, 720], [1105, 785], [1140, 860],
        [1110, 950], [1120, 1080], [200, 1080], [145, 990], [125, 890],
        [120, 790], [125, 690], [135, 590], [140, 490], [165, 400],
        [200, 350],
    ],
    dtype=np.int32,
)


def segment_xiaoxin(image: np.ndarray) -> np.ndarray:
    """在完整身体轮廓内提取蜡笔小新，并排除妈妈、背景和左侧玩具。"""
    actual_size = (image.shape[1], image.shape[0])
    if actual_size != XIAOXIN_SIZE:
        raise ValueError(
            "v2 图片应为 1920x1080，实际为 {0}x{1}".format(*actual_size)
        )

    region = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(region, [XIAOXIN_FULL_BODY_POLYGON], 255)

    blue, green, red = cv2.split(image)
    red16 = red.astype(np.int16)
    green16 = green.astype(np.int16)
    blue16 = blue.astype(np.int16)

    red_shirt = (red16 > 130) & (green16 < 130) & (blue16 < 105)
    red_shirt &= (red16 - green16) > 55
    red_shirt &= (red16 - blue16) > 30

    skin_normal = (red16 > 205) & (green16 > 125) & (blue16 > 125)
    skin_normal &= (red16 - green16) >= 30
    skin_normal &= (red16 - green16) <= 125
    skin_normal &= (green16 - blue16) >= 8
    skin_normal &= (green16 - blue16) <= 50

    skin_shadow = (red16 > 215) & (green16 > 115) & (blue16 > 105)
    skin_shadow &= (red16 - green16) >= 85
    skin_shadow &= (red16 - green16) <= 155
    skin_shadow &= (green16 - blue16) >= 5
    skin_shadow &= (green16 - blue16) <= 45

    skin = skin_normal | skin_shadow
    dark_hair = (red16 < 75) & (green16 < 75) & (blue16 < 75)
    seed = (red_shirt | skin | dark_hair) & (region > 0)
    seed_mask = seed.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    connected = cv2.morphologyEx(
        seed_mask, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    count, labels = cv2.connectedComponents(connected, connectivity=8)
    if count <= 1:
        raise RuntimeError(
            "没有找到蜡笔小新前景，请检查颜色条件和完整轮廓。"
        )

    red_labels = labels[red_shirt & (region > 0)]
    if red_labels.size == 0:
        raise RuntimeError(
            "没有找到小新红衣服区域，请检查红色衣服阈值。"
        )

    red_counts = np.bincount(red_labels.ravel(), minlength=count)
    red_counts[0] = 0
    main_label = int(np.argmax(red_counts))
    mask = np.where(labels == main_label, 255, 0).astype(np.uint8)

    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    yy, xx = np.indices(mask.shape)
    brown_hair = (red16 > 60) & (red16 < 195)
    brown_hair &= (green16 > 30) & (green16 < 150) & (blue16 < 110)
    brown_hair &= (red16 - green16) > 20
    brown_hair &= green16 >= blue16
    right_background = (xx > 650) & (yy < 900)
    mask[brown_hair & right_background] = 0

    keep_color = skin | dark_hair | red_shirt
    left_background = (xx < 240) & (yy > 250)
    top_background = (xx < 620) & (yy < 300)
    mask[left_background & (~keep_color)] = 0
    mask[top_background & (~keep_color)] = 0

    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(mask) < 5000:
        raise RuntimeError(
            "蜡笔小新掩膜面积过小，请检查完整轮廓坐标。"
        )
    return mask


def segment_with_grabcut(
    image: np.ndarray, rect: Optional[Rect], iterations: int
) -> np.ndarray:
    """对非内置图片使用普通 GrabCut 作为回退方案。"""
    if rect is None:
        raise SystemExit("处理非内置图片时必须提供 --rect x,y,w,h")

    x, y, width, height = rect
    rect_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        rect_mask,
        (x, y, width, height),
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    binary = np.where(
        (rect_mask == cv2.GC_FGD) | (rect_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    binary = polish_mask(binary, open_size=5)
    binary = keep_largest_component(binary)
    return fill_holes(binary)


def make_preview(image: np.ndarray, mask: np.ndarray, background: int) -> np.ndarray:
    """在指定底色上预览透明分割结果。"""
    preview = np.full_like(image, background)
    preview[mask > 0] = image[mask > 0]
    return preview


def make_boxed_image(
    image: np.ndarray, mask: np.ndarray, label: str
) -> np.ndarray:
    """在原图上绘制目标边界框和名称。"""
    boxed = image.copy()
    x, y, width, height = cv2.boundingRect(mask)

    cv2.rectangle(
        boxed,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 255, 0),
        3,
    )

    text_y = max(32, y - 10)
    cv2.putText(
        boxed,
        label,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return boxed


def make_contour_result(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """生成压暗背景、突出目标并绘制绿色轮廓的结果图。"""
    dark = (image.astype(np.float32) * 0.22).astype(np.uint8)
    soft_alpha = cv2.GaussianBlur(mask, (5, 5), 0)
    foreground = cv2.bitwise_and(image, image, mask=soft_alpha)
    weight = cv2.cvtColor(soft_alpha, cv2.COLOR_GRAY2BGR).astype(
        np.float32
    ) / 255.0
    visual = (
        foreground.astype(np.float32) * weight
        + dark.astype(np.float32) * (1.0 - weight)
    ).astype(np.uint8)

    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    cv2.drawContours(visual, contours, -1, (0, 255, 0), 3)
    return visual


def save_result(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    algorithm: str,
) -> None:
    """保存透明图、预览图、目标轮廓图、目标框图和 JSON 数据。"""
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = cv2.GaussianBlur(mask, (5, 5), 0)
    blue, green, red = cv2.split(image)
    transparent = cv2.merge([blue, green, red, alpha])

    x, y, width, height = cv2.boundingRect(mask)
    if width <= 0 or height <= 0:
        raise RuntimeError("分割掩膜中没有有效前景。")

    padding = 8
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)

    crop_box = (x1, y1, x2, y2)
    cropped_transparent = transparent[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    cropped_image = image[y1:y2, x1:x2]

    white_path = output_path.with_name(output_path.stem + "_preview.png")
    black_path = output_path.with_name(output_path.stem + "_black.png")
    mask_path = output_path.with_name(output_path.stem + "_mask.png")
    boxes_path = output_path.with_name(output_path.stem + "_boxes.png")
    result_image_path = output_path.with_name(output_path.stem + "_result.jpg")
    json_path = output_path.with_name(output_path.stem + "_result.json")

    imwrite_image(output_path, cropped_transparent)
    imwrite_image(
        white_path, make_preview(cropped_image, cropped_mask, 255)
    )
    imwrite_image(
        black_path, make_preview(cropped_image, cropped_mask, 0)
    )
    imwrite_image(mask_path, cropped_mask)
    imwrite_image(
        boxes_path, make_boxed_image(image, mask, output_path.stem)
    )
    imwrite_image(result_image_path, make_contour_result(image, mask))

    metadata = {
        "input_size": [int(image.shape[1]), int(image.shape[0])],
        "output_size": [
            int(cropped_transparent.shape[1]),
            int(cropped_transparent.shape[0]),
        ],
        "crop_box": [int(value) for value in crop_box],
        "target_box": [int(value) for value in (x, y, width, height)],
        "foreground_area": int(cv2.countNonZero(mask)),
        "algorithm": algorithm,
        "result_image": str(result_image_path),
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[完成] 透明背景图：{0}".format(output_path))
    print("       白底预览图：{0}".format(white_path))
    print("       目标轮廓图：{0}".format(result_image_path))
    print("       目标框图：{0}".format(boxes_path))
    print("       掩膜图片：{0}".format(mask_path))


def resolve_image_path(image_arg: str, image_dir: Path) -> Path:
    """把相对图片名解析到默认图片目录。"""
    image_path = Path(image_arg).expanduser()
    if image_path.is_absolute():
        return image_path
    return image_dir / image_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用专用算法分割奶龙和蜡笔小新。"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="输入图片名称或路径，例如 R-C.jpg",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="图片目录，默认为 ~/exam/opencv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出透明 PNG 路径；省略时按图片自动命名",
    )
    parser.add_argument(
        "--rect",
        type=parse_rect,
        help="兼容参数；非内置图片会使用该矩形运行普通 GrabCut",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="普通 GrabCut 的迭代次数，内置图片使用各自固定参数",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="兼容原命令；本程序默认不打开窗口",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = resolve_image_path(args.image, args.image_dir.expanduser())
    image = imread_color(image_path)
    if image is None:
        raise SystemExit("无法读取图片：{0}".format(image_path))

    image_name = image_path.name
    if image_name == IMAGE_NAMES["nailong"]:
        mask = segment_nailong(image)
        algorithm = "奶龙专用颜色阈值"
        default_prefix = "nailong"
    elif image_name == IMAGE_NAMES["xiaoxin"]:
        mask = segment_xiaoxin(image)
        algorithm = "蜡笔小新颜色与连通区域"
        default_prefix = "xiaoxin"
    else:
        mask = segment_with_grabcut(image, args.rect, args.iterations)
        algorithm = "普通 GrabCut"
        default_prefix = image_path.stem

    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / (default_prefix + ".png")
    elif not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    save_result(image, mask, output_path, algorithm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
