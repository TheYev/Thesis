# import cv2
# from ultralytics import YOLO
# import random


# def process_video_with_traking(model, input_video_path, show_video=True, save_video=False, output_video_path="output_video.mp4"):
#     cap = cv2.VideoCapture(input_video_path)
    
#     if not cap.isOpened():
#         raise Exception("Error: Could not open video.")
    
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_with = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     if save_video:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_with, frame_height))
        
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=640, verbose=False, tracker="botsort.yaml")
        
#         if results[0].boxes.id != None:
#             boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
#             ids = results[0].boxes.id.cpu().numpy().astype(int)
            
#             for box, id in zip(boxes, ids):
#                 random.seed(int(id))
#                 color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
#                 cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
#                 cv2.putText(
#                     frame,
#                     f"Id:{id}",
#                     (box[0], box[1]),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 255),
#                     2,
#                 )
                
#         if show_video:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_with, frame_height))
#             out.write(frame)
            
#         if show_video:
#             frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#             cv2.imshow("frame", frame)
            
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
    
#     cap.release()
#     if save_video:
#         out.release()
        
#     cv2.destroyAllWindows()

# model = YOLO("C:/Users/thedi/OneDrive/Desktop/d/runs/detect/train3/weights/best.pt")
# model.fuse()
# process_video_with_traking(model, "test_video.mp4", show_video=True, save_video=False)







import cv2
from ultralytics import YOLO
import random

# add rapam model
def process_video_with_tracking(input_video_path, show_video=True, save_video=False, output_video_path="output_video.mp4"):
    model = YOLO("C:/Users/thedi/OneDrive/Desktop/d/runs/detect/train3/weights/best.pt")
    model.fuse()
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    class_names = model.names  # Отримуємо словник класів

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, iou=0.4, conf=0.3, persist=True, imgsz=640, verbose=False, tracker="botsort.yaml")
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Отримуємо класи об'єктів
            confidences = results[0].boxes.conf.cpu().numpy()  # Отримуємо ймовірності

            for box, id, cls, conf in zip(boxes, ids, classes, confidences):
                class_name = class_names.get(cls, "Unknown")  # Отримуємо назву класу
                confidence = round(conf * 100, 2)  # Перетворюємо на відсотки
                
                random.seed(int(id))  
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    frame,
                    f"{class_name} {float('{:.2f}'.format(confidence))}%",
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                
        if show_video:
            cv2.imshow("frame", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
            
        if save_video:
            out.write(frame)
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()


# model = YOLO("C:/Users/thedi/OneDrive/Desktop/d/runs/detect/train3/weights/best.pt")
# model.fuse()
# process_video_with_tracking(model, "test_video.mp4", show_video=True, save_video=True)








# import cv2
# from ultralytics import YOLO

# model = YOLO("C:/Users/thedi/OneDrive/Desktop/d/runs/detect/train3/weights/best.pt")
# cap = cv2.VideoCapture("test_video.mp4")
# print("start")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)
#     annotated_frame = results[0].plot()

#     cv2.imshow("Military Detection", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("end")