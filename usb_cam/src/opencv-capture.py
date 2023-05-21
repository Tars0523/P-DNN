#!/usr/bin/env python

"""
CV2 video capture example from Pure Thermal 1
"""

import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import datetime
import imutils
import numpy as np
import sys
sys.path.append("/home/jiwoo/Documents/USB_CAM_/src/usb_cam/src")
import time
from centroidtracker import CentroidTracker
from cv_bridge import CvBridge
#class Robot:
#    def __init__(self):
        
#        self.rate = rospy.Rate(10)
#        self.bridge = CvBridge()

#        # Publisher to move robot
#        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
#        self.twist = Twist()

        # Image subscriber to read data from camera
        # img_sub = rospy.Subscriber('/camera/image_raw', Image, self.process_img)
bridge = CvBridge()
robot = Twist()
        
KNOWN_DISTANCE = 45
PERSON_WIDTH = 16

confThreshold =0.5
nmsThreshold= 0.2

labelsPath = '/home/jiwoo/Documents/USB_CAM_/src/usb_cam/src/coco.names'
weightsPath = '/home/jiwoo/Documents/USB_CAM_/src/usb_cam/src/yolov4-tiny.weights'
configPath = '/home/jiwoo/Documents/USB_CAM_/src/usb_cam/src/yolov4-tiny.cfg'

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


LABELS  =[]
with open(labelsPath, "r") as f:
	LABELS  = f.read().strip("\n").split("\n")
	

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

net =cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) 
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


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
 

def generate_boxes_confidences_classids(layerOutputs, H, W, confThreshold):
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

    
def tarck__():
	
	#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) 
	#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	#image =cv2.imread(image_pth)
	#(H, W) = image.shape[:2]
	
	cap =cv2.VideoCapture(2)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	
	ref = cv2.imread("/home/jiwoo/Documents/USB_CAM_/src/usb_cam/src/image14.png")
	person_data = object_detect(ref)
	refsize = person_data[0][1]


	# finding focal length 
	focal_person = real_length(KNOWN_DISTANCE, PERSON_WIDTH, refsize)
	
	rospy.init_node('robot', anonymous=True)
	pub_1 = rospy.Publisher('/cmd_vel',Twist, queue_size=1)
	pub_2 = rospy.Publisher('camera',Image,queue_size=1)
	

	
	while not rospy.is_shutdown():
		ret, image =cap.read()
		(H, W) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)

		net.setInput(blob)
		start = time.time()

		layerOutputs = net.forward(ln)
		end = time.time()
	    
	   
		rects =[]
		boxes, confidences, classIDs = generate_boxes_confidences_classids(layerOutputs, H, W, 0.5)
	    
	    
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	    
	   
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				rects.append((x, y, w, h))
		else:
			robot.angular.z=0
			robot.linear.x=0
			pub_1.publish(robot)

		    

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
			print(distance_id) 
			
   
			## ID 0
			firstid=objects[0]
			Fcenter=((firstid[0]+firstid[2]/2),((firstid[1]+firstid[3])/2))
			Hid=firstid[3]
			
			

			Hidx = 1 - Hid/440
			if (Hid>200 and Hid<410):
				robot.linear.x=Hidx*1.2
				pub_1.publish(robot)
	
			elif (Hid >450):
				robot.linear.x=-0.2
				pub_1.publish(robot)
	
			# 640
			Fidx = (320 - Fcenter[0]) /(400) # -3/4 ~ 3/4z
			if (Fcenter[0]<=640 and Fcenter[0]>=0): 
				robot.angular.z = Fidx*0.6
				pub_1.publish(robot)
    
    
	
			rospy.loginfo("linear vel : %f",robot.linear.x);
			rospy.loginfo("angluar vel : %f",robot.angular.z );
			pub_1.publish(robot)
		



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
		pub_2.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
	    # press "Q" to stop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

       
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    try:	tarck__()
  
    except rospy.ROSInterruptException:
        pass






