from ultralytics import YOLO



# 对文件夹ultralytics\assets里面所有图片做推理
if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(
        data=r"hulu.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=2,
    )