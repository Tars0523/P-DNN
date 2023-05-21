# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
       # 유일한 오브젝트 ID를 초기화하고, 주어진 오브젝트 ID를 해당 centroid와 연관 지을 수 있는
        # ordered dictionary를 초기화합니다. 이 ordered dictionary에는 각 오브젝트 ID와 해당 centroid가
        # 몇 번 연속해서 "사라졌는지"에 대한 정보가 포함됩니다.
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # 주어진 오브젝트가 추적에서 "사라진" 표시를 받을 수 있는 최대 연속 프레임 수를 저장합니다. 
        # 이 수치를 초과하면 해당 오브젝트는 tracking에서 제거됩니다.
        self.maxDisappeared = maxDisappeared

        # 두 centroid 사이의 최대 거리를 저장합니다. 
         # 이 최대 거리를 초과하면 해당 오브젝트를 "사라졌다"고 표시하기 시작합니다.
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
         # 오브젝트를 등록할 때 사용할 수 있는 다음 유일한 object ID를 사용하여
         # centroid를 저장합니다.
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  # CHANGE
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # 오브젝트 ID를 등록 해제하려면 각 ordered dictionary에서 해당 오브젝트 ID를 삭제합니다.
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  # CHANGE

    def update(self, rects):
       # 입력 bounding box rectangle 리스트가 비어있는지 확인합니다.
        # is empty
        if len(rects) == 0:
            # 이미 추적 중인 모든 오브젝트를 반복하고 해당 오브젝트를 "사라졌다"고 표시합니다.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

               # 주어진 오브젝트가 "사라진" 상태로 연속된 프레임 수가 최대 수를 초과하면 해당 오브젝트를 추적에서 제거합니다.
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

           #centroid 또는 tracking 정보가 없으므로 일찍 반환합니다.
            # return self.objects
            return self.bbox

       # 현재 frame의 input centroids를 초기화합니다.
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
       # bounding box rectangle을 반복합니다.
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
           # bounding box 좌표를 사용하여 중심 좌표를 계산합니다.
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])  # CHANGE

         # 객체를 추적하지 않는 경우 입력 중심점을 가져와 각각 등록합니다.
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE

       # 그렇지 않으면, 현재 객체를 추적 중이므로 입력 중심점을 기존 객체 중심점에 매칭해야 합니다.
        # centroids
        else:
            # 객체 ID와 해당 중심점을 가져옵니다.
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

             # 객체 중심점과 입력 중심점 간의 거리를 계산합니다.
            # 목표는 입력 중심점을 기존 객체 중심점에 매칭하는 것입니다.
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

           # 이 매칭을 수행하려면 (1) 각 행에서 가장 작은 값을 찾고 (2) 최소 값을 기준으로 행 인덱스를 정렬하여
            # 최소 값을 *front*에 둔 인덱스 목록을 정렬해야 합니다.
            # list
            rows = D.min(axis=1).argsort()

           # 다음으로, 이전에 계산된 행 인덱스 목록을 사용하여 열에서 유사한 과정을 수행합니다.
            # 열에서 가장 작은 값을 찾고, 이전에 계산된 행 인덱스 목록을 사용하여 정렬합니다.
            cols = D.argmin(axis=1)[rows]

           # 객체를 업데이트, 등록 또는 등록 해제해야 하는지 여부를 결정하려면,
            # 이미 검사한 행과 열 인덱스를 추적해야 합니다.
            usedRows = set()
            usedCols = set()

           # (행, 열) 인덱스 튜플의 조합을 순환합니다.
            # tuples
            for (row, col) in zip(rows, cols):
                # 이전에 행 또는 열 값을 이미 검사한 경우 무시합니다.
                if row in usedRows or col in usedCols:
                    continue

                # 중심점 간 거리가 최대 거리보다 큰 경우 두 중심점을 동일한 객체에 매핑하지 않습니다.
                if D[row, col] > self.maxDistance:
                    continue

                #그렇지 않으면 현재 행의 객체 ID를 가져와서,새로운 중심점을 설정하고 사라진 프레임 수를 재설정합니다.
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0

                 # 각각의 행 및 열 인덱스를 조사했음을 나타냅니다.
                usedRows.add(row)
                usedCols.add(col)

           # 아직 조사하지 않은 행 및 열 인덱스를 계산합니다.
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

             # 객체 중심점의 수가 입력 중심점의 수와 동일하거나 더 많은 경우
             # 일부 객체가 사라졌는지 확인해야합니다.
            if D.shape[0] >= D.shape[1]:
                 # 사용하지 않은 행 인덱스를 순환합니다.
                for row in unusedRows:
                   # 해당 행 인덱스에 대한 객체 ID를 가져오고 사라진 프레임 수를 증가시킵니다.
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                   # 객체가 "사라졌다"로 표시된 연속 프레임 수가
                    # 객체를 등록 취소해야 할 정도로 많은지 확인합니다.
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # 그렇지 않으면, 입력 중심점의 수가 기존 객체 중심점의 수보다 큰 경우
            # 각 새로운 입력 중심점을 추적 가능한 객체로 등록해야합니다.
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        #추적 가능한 객체 집합을 반환합니다.
        # return self.objects
        return self.bbox #근데 나는 bbox가져가서 할거