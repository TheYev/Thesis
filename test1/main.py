import cv2


print("Start")

cap = cv2.VideoCapture('video.mp4')
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Frame', frame)
    # cv2.resizeWindow('Frame', 800, 600)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
