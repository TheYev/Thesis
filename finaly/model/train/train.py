if __name__ == "__main__":
    import torch
    from ultralytics import YOLO

    model = YOLO('yolov8x.pt')
    model.train(
        data='C:/Users/thedi/OneDrive/Desktop/d/finaly/model/train/dataset/data.yaml', 
        epochs=20, 
        imgsz=640, 
        batch=16, 
        model="yolov8x.pt",
        device='0',
        )

# print(torch.cuda.is_available())  # Повинно вивести True
# print(torch.cuda.device_count())  # Кількість доступних GPU
# print(torch.cuda.get_device_name(0))  # Назва GPU