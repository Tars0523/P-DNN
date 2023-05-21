import cv2
import datetime
import imutils
import numpy as np

import time
from centroidtracker import CentroidTracker

KNOWN_DISTANCE = 15 
PERSON_WIDTH = 16 

confThreshold =0.5
nmsThreshold= 0.2

labelsPath = 'coco.names'
weightsPath = 'yolov4-tiny.weights'
configPath = 'yolov4-tiny.cfg'

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
#tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

LABELS  =[]
with open(labelsPath, "r") as f:
    LABELS  = f.read().strip("\n").split("\n")
    

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

net =cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #gpu용
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #cpu용
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#image =cv2.imread(image_pth)
#(H, W) = image.shape[:2]
cap =cv2.VideoCapture(0)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def object_detect(image):
    classes, scores, boxes = model.detect(image, confThreshold, nmsThreshold)
    data_list =[]
    person_list=[]
    person_count = 0
    for (classid, score, box) in zip(classes, scores, boxes):
        if classid ==0: 
            data_list.append([LABELS[classid], box[2], (box[0], box[1])])
            person_list.append([LABELS[classid], box])
        print(data_list)
    return data_list 
 
# 입력 이미지로부터 blob을 생성하고 YOLO 객체 검출기에 전방향 패스를 수행하여바운딩 박스와 관련된 확률을 제공합니다.
def generate_boxes_confidences_classids(layerOutputs, H, W, confThreshold): #감지된 개체의 경계 상자, 신뢰도 및 클래스 ID를 추출
	boxes = [] 
	confidences = []
	classIDs = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > confThreshold:
				if LABELS[classID] != "person":
					continue
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	return boxes, confidences, classIDs
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



#FPS
# fps_start_time = datetime.datetime.now()
# fps = 0
# total_frames = 0
# writer =None

# frames_count =0

# fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
# fps = 10.0
# capture_size = (int(cap.get(3)), int(cap.get(4)))

# out = cv2.VideoWriter("output_yolo_pidft_v3.avi", fourcc_codec, fps, capture_size)

while True:
    ret, image =cap.read()
    #image = imutils.resize(image, width=600)
    # total_frames = total_frames + 1
    
    if not ret:
        break
    
    (H, W) = image.shape[:2]
    

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    	swapRB=True, crop=False)
    
    net.setInput(blob)
    start = time.time()
    
    layerOutputs = net.forward(ln)
    # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
    end = time.time()
    
    # 타이밍정보(주석처리해논건데 필요없을듯)
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # 감지된 바운딩 박스, 신뢰도 및 클래스 ID의 목록을 초기화합니다.
    rects =[]
    boxes, confidences, classIDs = generate_boxes_confidences_classids(layerOutputs, H, W, 0.5)
    
    # 약한, 겹치는 바운딩 박스를 억제하기 위해 비최대 억제를 적용합니다.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    # 1개라도 존재시
    if len(idxs) > 0:
        # 인덱스 반복
        for i in idxs.flatten(): #.flatten =>다차원 배열 공간을 1차원으로 평탄화
            # 박스값 추출
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            rects.append((x, y, w, h))
            

    #tracker
    objects = tracker.update(rects)
    print(objects)
    objectId_ls =[]
    distance_id=[]
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        objectId_ls.append(objectId)

        cv2.rectangle(image, (x1, y1), (x1+x2, y1+y2), (0,255,0), 2)
        text = "ID:{}".format(objectId)
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        box_center = (int(x1 + x2/2), int(y1 + y2/2))
        cv2.circle(image, box_center, 5, (0, 0, 255), -1)
        distance = distance_cal(focal_person, PERSON_WIDTH, y1)
        realdistance=round(distance*2.54,2)
        distance_id.append(realdistance)
        cv2.putText(image, f'Distance: {round(distance*2.54,2)}cm', (x1+5,y1+13), cv2.FONT_ITALIC, 0.5, (0,0,0),1)
        print(f'Distance: {realdistance}cm')   
        print(distance_id) 


    #FPS
    # fps_end_time = datetime.datetime.now()
    # time_diff = fps_end_time - fps_start_time
    # if time_diff.seconds == 0:
    #     fps = 0.0
    # else:
    #     fps = (total_frames / time_diff.seconds)

    # fps_text = "FPS: {:.2f}".format(fps)

    # cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 0), 2)
    
    #write
    # out.write(image)
    # show the output image
    cv2.imshow("Image", image)
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #writer.write(image)
    
print("list of all object id:", objectId_ls)
#writer.release()
cap.release()
# out.release()

cv2.destroyAllWindows()