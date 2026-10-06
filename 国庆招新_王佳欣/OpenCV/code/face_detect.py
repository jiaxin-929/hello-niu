#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 OpenCV Haar 级联分类器进行人脸检测。

直接运行时会依次处理：
    单人半身.jpg -> outputs/face_single.png
    多人合影.jpg -> outputs/face_group.png

也可以使用 --image 单独处理一张图片。
程序默认无窗口运行，适合 Ubuntu 虚拟机和 VS Code Remote-SSH。
"""

from __future__ import annotations

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


DEFAULT_IMAGE_DIR = Path.home() / "exam" / "opencv"
DEFAULT_OUTPUT_DIR = DEFAULT_IMAGE_DIR / "outputs"
FaceBox = Tuple[int, int, int, int]

# 肤色过滤阈值，用于排除纽扣、衣领、手臂和衣服图案等误检。
MIN_FACE_AREA = 3500
MIN_SKIN_RATIO = 0.25
MIN_CENTER_SKIN_RATIO = 0.62
MIN_SIDE_SKIN_RATIO = 0.35

# 无人值守运行时的参考参数。
DEFAULT_TASKS: List[Dict[str, object]] = [
    {
        "image": "单人半身.jpg",
        "output": "face_single.png",
        "scale": 1.05,
        "neighbors": 6,
        "min_size": 60,
    },
    {
        "image": "多人合影.jpg",
        "output": "face_group.png",
        "scale": 1.05,
        "neighbors": 6,
        "min_size": 50,
    },
]


def imread_color(path: Path) -> Optional[np.ndarray]:
    """读取彩色图片，并兼容中文路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_image(path: Path, image: np.ndarray) -> None:
    """保存图片，并兼容中文路径。"""
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


def create_detector() -> cv2.CascadeClassifier:
    """创建 OpenCV 正面人脸 Haar 分类器。"""
    classifier_class = getattr(cv2, "CascadeClassifier", None)
    if classifier_class is None and hasattr(cv2, "objdetect"):
        classifier_class = getattr(cv2.objdetect, "CascadeClassifier", None)

    if classifier_class is None:
        module_version = getattr(cv2, "__version__", "未知")
        module_file = getattr(cv2, "__file__", "未知")
        has_objdetect = hasattr(cv2, "objdetect")
        raise SystemExit(
            "当前 cv2 模块缺少 CascadeClassifier，无法进行 Haar 人脸检测。\n"
            "cv2 版本：{0}\n"
            "cv2 路径：{1}\n"
            "cv2.objdetect 是否存在：{2}\n"
            "OpenCV 4 使用 cv2.CascadeClassifier，"
            "OpenCV 5 可能使用 cv2.objdetect.CascadeClassifier。\n"
            "如果两种路径都不存在，请在已激活的 opencv_env 中执行：\n"
            "python -m pip uninstall -y cv2 opencv-python "
            "opencv-python-headless opencv-contrib-python\n"
            "python -m pip install --force-reinstall --no-cache-dir "
            "opencv-python==4.8.1.78 numpy==1.24.4\n"
            "安装完成后重新运行本程序。".format(
                module_version, module_file, has_objdetect
            )
        )

    cascade_filename = "haarcascade_frontalface_default.xml"
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / cascade_filename)

    module_file = getattr(cv2, "__file__", None)
    if module_file:
        candidates.append(
            Path(module_file).resolve().parent / "data" / cascade_filename
        )
    candidates.append(
        Path("/usr/share/opencv4/haarcascades") / cascade_filename
    )

    cascade_path = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None,
    )
    if cascade_path is None:
        raise SystemExit(
            "找不到 Haar 人脸分类器文件，请重新安装 opencv-python。"
        )

    detector = classifier_class(str(cascade_path))
    if detector.empty():
        raise SystemExit(
            "无法加载 Haar 人脸分类器：{0}".format(cascade_path)
        )
    return detector


def skin_ratio(
    image: np.ndarray, x: int, y: int, width: int, height: int
) -> float:
    """计算指定矩形区域内的肤色像素比例。"""
    if width <= 0 or height <= 0:
        return 0.0

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    skin_1 = cv2.inRange(
        hsv,
        np.array([0, 40, 80], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8),
    )
    skin_2 = cv2.inRange(
        hsv,
        np.array([170, 40, 80], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )
    skin = cv2.bitwise_or(skin_1, skin_2)
    return cv2.countNonZero(skin) / float((x2 - x1) * (y2 - y1))


def is_valid_face(
    image: np.ndarray, x: int, y: int, width: int, height: int
) -> bool:
    """使用面积、宽高比和肤色分布过滤误检。"""
    if width * height < MIN_FACE_AREA:
        return False

    aspect = width / float(height)
    if aspect < 0.72 or aspect > 1.38:
        return False

    if skin_ratio(image, x, y, width, height) < MIN_SKIN_RATIO:
        return False

    center_ratio = skin_ratio(
        image,
        x + width // 4,
        y + height // 4,
        width // 2,
        height // 2,
    )
    if center_ratio < MIN_CENTER_SKIN_RATIO:
        return False

    left_ratio = skin_ratio(image, x, y, width // 2, height)
    right_ratio = skin_ratio(
        image,
        x + width // 2,
        y,
        width - width // 2,
        height,
    )
    if min(left_ratio, right_ratio) < MIN_SIDE_SKIN_RATIO:
        return False

    return True

def sort_faces(faces: Sequence[Sequence[int]]) -> List[FaceBox]:
    """按照从上到下、从左到右的顺序排列人脸框。"""
    if len(faces) == 0:
        return []
    return sorted(
        [tuple(map(int, face)) for face in faces],
        key=lambda face: (face[1] // 50, face[0]),
    )


def detect_one(
    image_path: Path,
    output_path: Path,
    detector: cv2.CascadeClassifier,
    scale: float,
    neighbors: int,
    min_size: int,
) -> int:
    """检测一张图片中的人脸并保存结果，返回人脸数量。"""
    image = imread_color(image_path)
    if image is None:
        raise RuntimeError("无法读取图片：{0}".format(image_path))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    candidates = detector.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=(min_size, min_size),
    )

    faces: List[FaceBox] = []
    for x, y, width, _ in candidates:
        normalized_height = int(width * 1.15)
        if y + normalized_height > image.shape[0]:
            continue
        if is_valid_face(image, x, y, width, normalized_height):
            faces.append((x, y, width, normalized_height))
    faces = sort_faces(faces)

    result = image.copy()
    for index, (x, y, width, height) in enumerate(faces, start=1):
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            result,
            "face {0}".format(index),
            (x, max(28, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path = output_path.with_suffix(".png")
    imwrite_image(output_path, result)

    reference_path = output_path.with_name(
        "face_{0}_result.jpg".format(image_path.stem)
    )
    if reference_path.resolve() != output_path.resolve():
        imwrite_image(reference_path, result)

    metadata = {
        "image": str(image_path),
        "output": str(output_path),
        "reference_result": str(reference_path),
        "candidate_count": int(len(candidates)),
        "parameters": {
            "scale": scale,
            "neighbors": neighbors,
            "min_size": min_size,
        },
        "face_count": len(faces),
        "faces": [
            {"box": [int(x), int(y), int(width), int(height)]}
            for x, y, width, height in faces
        ],
    }
    json_path = output_path.with_name(output_path.stem + "_result.json")
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "[完成] {0}：原始候选 {1} 个，过滤后检测到 {2} 张人脸".format(
            image_path.name, len(candidates), len(faces)
        )
    )
    print("       结果图：{0}".format(output_path))
    print("       参考命名：{0}".format(reference_path))
    print("       数据文件：{0}".format(json_path))
    if len(faces) == 0:
        print("       未检测到人脸，可调小 --neighbors 或 --min-size。")
    return len(faces)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Haar 分类器检测单人或合影中的人脸。"
    )
    parser.add_argument(
        "--image",
        help="只处理指定图片；省略时自动处理单人和合影两张素材",
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
        help="单图模式下的输出路径",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="图像金字塔缩放系数；单人 1.1，合影 1.05",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        help="邻近矩形数量阈值；漏检时可调小",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        help="最小人脸尺寸；合影建议 20",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="兼容原命令；本程序始终无窗口运行",
    )
    return parser


def build_tasks(args: argparse.Namespace) -> Sequence[Dict[str, object]]:
    """创建默认批量任务或用户指定的单图任务。"""
    if args.image is None:
        if args.output is not None:
            raise SystemExit("使用 --output 时必须同时指定 --image")
        tasks = []
        for default_task in DEFAULT_TASKS:
            task = dict(default_task)
            task["output"] = str(
                DEFAULT_OUTPUT_DIR / str(default_task["output"])
            )
            tasks.append(task)
        return tasks

    image_path = Path(args.image).expanduser()
    image_name = image_path.name
    default_task = next(
        (task for task in DEFAULT_TASKS if task["image"] == image_name),
        None,
    )

    if default_task is not None:
        # 内置图片固定使用参考参数，兼容旧命令但不会采用容易误检的宽松参数。
        scale = float(default_task["scale"])
        neighbors = int(default_task["neighbors"])
        min_size = int(default_task["min_size"])
        if args.output is None:
            output_path = DEFAULT_OUTPUT_DIR / str(default_task["output"])
        else:
            output_path = Path(args.output).expanduser()
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
    else:
        scale = args.scale if args.scale is not None else 1.05
        neighbors = args.neighbors if args.neighbors is not None else 6
        min_size = args.min_size if args.min_size is not None else 50
        if args.output is None:
            output_path = DEFAULT_OUTPUT_DIR / (image_path.stem + "_faces.png")
        else:
            output_path = Path(args.output).expanduser()
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path

    return [
        {
            "image": str(image_path),
            "output": str(output_path),
            "scale": scale,
            "neighbors": neighbors,
            "min_size": min_size,
        }
    ]

def main() -> int:
    args = build_parser().parse_args()
    detector = create_detector()
    image_dir = args.image_dir.expanduser()
    output_dir = DEFAULT_OUTPUT_DIR

    tasks = build_tasks(args)
    for task in tasks:
        scale = float(task["scale"])
        neighbors = int(task["neighbors"])
        min_size = int(task["min_size"])

        if scale <= 1.0:
            raise SystemExit("--scale 必须大于 1.0")
        if neighbors < 1:
            raise SystemExit("--neighbors 必须大于等于 1")
        if min_size < 1:
            raise SystemExit("--min-size 必须大于等于 1")

        image_value = Path(str(task["image"]))
        image_path = (
            image_value if image_value.is_absolute() else image_dir / image_value
        )
        output_value = Path(str(task["output"])).expanduser()
        output_path = (
            output_value
            if output_value.is_absolute()
            else Path.cwd() / output_value
        )

        detect_one(
            image_path=image_path,
            output_path=output_path,
            detector=detector,
            scale=scale,
            neighbors=neighbors,
            min_size=min_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())