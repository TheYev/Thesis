from ultralytics import YOLO

# Завантаження предтренованої моделі
model = YOLO("yolov8n.pt")

# Тренування на власному датасеті
model.train(data="dataset/data.yaml", epochs=50, imgsz=640, batch=16)
 
 
 
#Після завершення тренування збережеться новий файл ваг:
 

#Тестування моделі
#model = YOLO("runs/detect/train/weights/best.pt")  # Використовуємо нашу модель
#results = model("test_image.jpg", save=True)