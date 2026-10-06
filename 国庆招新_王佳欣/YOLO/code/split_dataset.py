from ultralytics import YOLO

# 加载训练好的葫芦模型
model = YOLO(r"D:\ultralytics-8.3.163\runs\detect\train9\weights\best.pt")

# 推理图片
results = model(r"D:\ROC\yolo\hulu\images\train\TZ1_No068(2).jpg")

for res in results:
    res.show()
    res.save("hulu_out3.jpg")