from ultralytics import YOLO
import cv2

model = YOLO('best.onnx')
image = 'c.jpg'

while True:

    results = model(image)
    result = results[0]

    annotated_frame = result.plot() 
    cv2.imwrite('output2.jpg',annotated_frame)
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
