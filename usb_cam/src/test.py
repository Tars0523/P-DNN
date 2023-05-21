import cv2
import numpy as np
import cv2, sys, os
import math
#라이다툴박스 네비게이션,

def diagonal(a,b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

if not (os.path.isfile('goturn.caffemodel') and os.path.isfile('goturn.prototxt')):
    errorMsg = '''
    Could not find GOTURN model in current directory.
    Please ensure goturn.caffemodel and goturn.prototxt are in the current directory
    '''

    print(errorMsg)
    sys.exit()
# 카메라로부터 입력 받기

cap = cv2.VideoCapture(0)

# 5초 대기
for i in range(5):
    ret, frame = cap.read()
    
    cv2.waitKey(1000)

# 새로운 프레임 읽기
ret, frame = cap.read()

# 영상 가운데 좌표
height, width, _ = frame.shape
center = (int(width/2), int(height/2))

# 가운데 물체 찾기
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 200, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
x = int(x + w/4)
y = int(y + h/4)
w = int(w/2)
h = int(h/2)
object_center = (int(x + w/2), int(y + h/2))

# KCF tracker 초기화

tracker = cv2.TrackerGOTURN_create()
tracker.init(frame, (x, y, w, h))

# 추적 시작
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 추적하기
    success, box = tracker.update(frame)
    
    # 추적 성공시, 추적된 물체 표시하기
    if success:
        x, y, w, h = [int(i) for i in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        a=(x,y)
        b=(x+w,y+h)
        d=diagonal(a,b)
        print(d)
        # 박스 중심점 그리기
        box_center = (int(x + w/2), int(y + h/2))
        
        # 빨간 점 그리기
        cv2.circle(frame, box_center, 5, (0, 0, 255), -1)
        
        # 2구역 나누기
        cv2.line(frame, (center[0], 0), (center[0], height), (255, 0, 0), 1)
        
         # Detect 메시지 출력
        cv2.putText(frame, "Detect!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if(d<200):
            cv2.putText(frame,"go straight",(50,90),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
        if box_center[0]< center[0]/2:
            cv2.putText(frame,"Left",(50,60),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
        elif box_center[0]>center[0]+center[0]/2: 
            cv2.putText(frame,"Right",(50,60),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
        
        if(d>350):
            cv2.putText(frame,"go Back",(50,90),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
        
            

    else:
        # No Detect 메시지 출력
        cv2.putText(frame, "No Detect", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    
    # 영상 출력하기
    winname = "test"
    cv2.namedWindow(winname)   # create a named window
    # cv2.moveWindow(winname, 40, 30)   # Move it to (40, 30)
    # cv2.resizeWindow(winname, 640, 1280);
    cv2.imshow(winname, frame)
    
    # 종료 조건
    if cv2.waitKey(1) == ord('q'):
        break

# 해제하기
cap.release()
cv2.destroyAllWindows()