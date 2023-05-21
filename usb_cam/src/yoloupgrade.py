import cv2
import numpy as np

KNOWN_DISTANCE = 15 
PERSON_WIDTH = 16 
 
CONFIDENCE_THRESHOLD = 0.5 
NMS_THRESHOLD = 0.3 

cam_width = 1280
cam_height = 720


COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
RED=(0,0,255)
BLACK =(0,0,0)
GREEN =(0,255,0)
WHITE=(255,255,255)
FONTS = cv2.FONT_ITALIC


yoloNet = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
tracker=cv2.TrackerKCF()


class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
    
model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def object_detect(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list =[]
    person_list=[]
    person_count = 0
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "%s : %f" % (class_names[classid[0]], score)

        if classid ==0:
            print(box)
            person_count += 1 
            # if person_count==1:
                
            person_label = f"Person {person_count}" 
            cv2.rectangle(image, box, GREEN, 2)
            cv2.putText(image, person_label, (box[0], box[1]-30), FONTS, 1, BLACK, 2)
            cv2.putText(image, label, (box[0], box[1]-10), FONTS, 0.5, RED, 2)

       
        if classid ==0: 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1])])
            person_list.append([class_names[classid[0]], box])
        print(data_list)
    return data_list  

def real_length (measured_distance, real_width, refwidth):
    real = (refwidth * measured_distance) / real_width

    return real

def distance_cal(length, real_object, widthframe):
    distance = (real_object * length) / widthframe
    return distance

ref = cv2.imread('image.png')

person_data = object_detect(ref)
refsize = person_data[0][1]

print(f"Person in pixels : {refsize}")

# finding focal length 
focal_person = real_length(KNOWN_DISTANCE, PERSON_WIDTH, refsize)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    data = object_detect(frame)
     
    for d in data:
        if d[0] =='person':
            distance = distance_cal(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
            
        cv2.putText(frame, f'Distance: {round(distance*2.54,2)}cm', (x+5,y+13), FONTS, 0.5, BLACK)

    cv2.imshow('CAM',frame)
    key = cv2.waitKey(1)
    

    if key ==ord('s'):
        break
    
cv2.destroyAllWindows()
cap.release()

