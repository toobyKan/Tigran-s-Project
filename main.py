from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video = cv2.VideoCapture("traffic.mp4") #

while (True):
    ret, frame = video.read()

    detect = model.track(frame, persist=True)

    frame_ = detect[0].plot()

    cv2.imshow("VIDEO", frame_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


