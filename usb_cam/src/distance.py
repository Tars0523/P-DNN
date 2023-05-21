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

# finding focal length 
focal_person = real_length(KNOWN_DISTANCE, PERSON_WIDTH, refsize)



