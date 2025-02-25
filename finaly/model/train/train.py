from ultralytics import YOLO

model = YOLO('yolov8x.pt')
model.train(data='C:/Users/thedi/OneDrive/Desktop/d/finaly/model/train/dataset/data.yaml', epochs=100, imgsz=640, batch=16, model="yolov8x.pt")
# import torch
# print(torch.cuda.is_available())  # Чи доступний CUDA
# print(torch.cuda.device_count())  # Кількість доступних GPU
# for i in range(torch.cuda.device_count()):
    # print(f"GPU {i}: {torch.cuda.get_device_name(i)}")