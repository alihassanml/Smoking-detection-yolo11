from ultralytics import YOLO
import cv2

model = YOLO('best.onnx')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break  

    results = model(frame)
    result = results[0]

    annotated_frame = result.plot() 
    cv2.imshow('YOLO Inference', annotated_frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
