from ultralytics import YOLO

# Завантаження предтренованої моделі
model = YOLO("yolov8m.pt")

# Тренування на власному датасеті
model.train(data="dataset/data.yaml", epochs=100, imgsz=640, batch=16, model="yolov8m.pt")
# print(model.val())
# Оцінка точності моделі
# model.test(data="C:/Users/thedi/OneDrive/Desktop/d/test2/dataset/data.yaml")
