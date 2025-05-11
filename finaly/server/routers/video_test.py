import cv2
from ultralytics import YOLO
import random
import os

def video_counter(video_dir_path):
    count = 0
    for file in os.listdir(video_dir_path):
        if file.endswith(".mp4"):
            count += 1
    return count

async def process_video_with_tracking(input_video_path, show_video=True, save_video=False):
    model = YOLO("C:/Users/thedi/OneDrive/Desktop/d/finaly/server/routers/best.pt")
    model.fuse()
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_dir_path = "C:/Users/thedi/OneDrive/Desktop/d/finaly/out_video"
    count_video = video_counter(video_dir_path)
    output_video_path=f"{video_dir_path}/output_video{count_video}.mp4"

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    class_names = model.names  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, iou=0.4, conf=0.3, persist=True, imgsz=640, verbose=False, tracker="botsort.yaml")
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)  
            confidences = results[0].boxes.conf.cpu().numpy()  

            for box, id, cls, conf in zip(boxes, ids, classes, confidences):
                class_name = class_names.get(cls, "Unknown")  
                confidence = round(conf * 100, 2)  
                
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
    return output_video_path


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