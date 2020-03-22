import cv2
from sort import *
import pandas as pd
import sys
import re
import time
import json
from datetime import datetime
import argparse

counter = 0
counted_vehicles = []
previous_vehicle_ids=[]
discarded_vehicle_ids =[]
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    
    parser.add_argument("--file", dest = 'file', help = "csv file path",
                        default = "1.csv", type = str)
        
    parser.add_argument("--video", dest = 'video', help = "video path",
                        default = "video.avi", type = str)                    
  
    
    return parser.parse_args()
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def overlap(bbox, roi):
	left = np.maximum(bbox[0],roi[0])
	top = np.maximum(bbox[1],roi[1])
	right=np.minimum(bbox[2],roi[2])
	bottom= np.minimum(bbox[3],roi[3])
	return  (left<right and top <bottom)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


tracker = Sort()
memory = {}
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
arg = arg_parse()
yolo_detection_fileName = arg.file
filename = arg.video
totalFrames = int(yolo_detection_fileName.split('_')[1])
#totalFrames = 377
print(totalFrames)
df = pd.read_csv(yolo_detection_fileName)
df = df[df.columns[1:]]

df = df.values.tolist()

data_dict = {}
all_data = []
calc_timestamps = [0.0]
for d in range(int(totalFrames)):
    data_dict[d] = []


for d in df:
    # print(d, d[0], d[1:])
    data_dict[int(d[0])].append(d)
    
(W, H) = (None, None)

cap = cv2.VideoCapture(filename)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps = ", fps)
for _i in range(len(data_dict.keys())):
    vehicles_crossed = 0
    calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    # print(data_dict[i][1:5])
    dets =[]
    print(_i)
    ret, frame = cap.read()
    if not ret:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    for row in data_dict[_i]:
        dets.append(row[1:6])
    print(dets)
    dets = np.asarray(dets)
    tracks = tracker.update(dets)
    lineOfInterest = [(int(0.1*W),int(0.45*H)),(int(0.73*W),int(0.45*H))]
    boxes = []
    indexIDs = []
    previous = memory.copy()
    memory = {}
    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]
    previous_vehicle_ids = list(previous.keys())
    counted_vehicles = list(set(counted_vehicles).difference(  (set(previous_vehicle_ids)|set(indexIDs)) - set(indexIDs) ))
    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)

                if intersect(p0, p1, lineOfInterest[0], lineOfInterest[1]):
                    counter += 1
                    vehicles_crossed += 1

                
            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1
    
    all_data.append({calc_timestamps[-1]:vehicles_crossed})
    overlay = frame.copy()
    cv2.line(frame,(lineOfInterest[0]),(lineOfInterest[1]),(255,0,0),2)
    cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
    cv2.imwrite("./sortresults/frame_{}.jpg".format(_i),frame)
    #print(all_data)
timeStamps =[]
count=[]
for ele in all_data:
    for i in ele:
        timeStamps.append(i)
        count.append(ele[i])
timeStamps = [int(str(x).split('.')[0]) for x in timeStamps]
_timeStamps = [x/1000 for x in timeStamps]
print(_timeStamps)

count_dict= {}
for time,c in zip(_timeStamps,count):
    #print(int(time),c)    
    if int(time) not in count_dict:
        count_dict[int(time)]=c
    elif int(time) in count_dict:
        count_dict[int(time)] += c
print(count_dict)

a = yolo_detection_fileName.split('_')[0]
day = a[1:3]
hour = a[4:6]
minute = a[7:9]
sec = a[10:12]
p = '19-09-{} {}:{}:{}'.format(day,hour,minute,sec)
p = datetime.strptime(p,'%y-%m-%d %H:%M:%S')
c = {}
for keys in count_dict:
    c[str(datetime.fromtimestamp(datetime.timestamp(p)+ int(keys)))] = count_dict[keys]
print(sum(c.values()))

df = pd.DataFrame(list(c.items()),columns=['Time Elaspsed','Vehicle Count'])
print(yolo_detection_fileName)
print(yolo_detection_fileName.split('_')[0])
df.to_csv('result/'+yolo_detection_fileName.split('_')[0]+"_SORT.csv",index=False)

print('...completed')
