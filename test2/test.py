from ultralytics import YOLO

#Тестування моделі
model = YOLO("../runs/detect/train/weights/best.pt")  # Використовуємо нашу модель
# model.fuse()# Об'єднання моделі
result = model.predict("1.jpg", imgsz=640, show_boxes=True)  # Тестування моделі
# result.show()  # Відображення результатів
# for r in result:
    # print(r.boxes)

for r in result:
    print(r.boxes)
    print(r.masks)
    print(r.keypoints)
    print(r.probs)
    print(r.obb)
    r.show()
# boxes = result.boxes  # Boxes object for bounding box outputs
# masks = result.masks  # Masks object for segmentation masks outputs
# keypoints = result.keypoints  # Keypoints object for pose outputs
# probs = result.probs  # Probs object for classification outputs
# obb = result.obb  # Oriented boxes object for OBB outputs
# result.show()  # display to screen
# result.save(filename="result.jpg")

# print(boxes, masks, keypoints, probs, obb)  # print results to screen
# print(result)