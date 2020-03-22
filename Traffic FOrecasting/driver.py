import cv2
from detect import Detect
from detect import arg_parse
import numpy as np
import time
import pandas as pd
import os
from subprocess import call
import gc
import os
from darknet import Darknet


if __name__ == '__main__':
    
    arg = arg_parse()
    print("loading model")
    model = Darknet(arg.cfgfile)
    model.load_weights(arg.weightsfile)
    print("model loaded")
    BatchSize = 8 #frames
    lineOfInterest = [(10,250),(500,250)]
    f = arg.video
    #video_name = 'New export_r01_RUPALI-Sq-AP-RAM_MANDIR-OC-01(209)_y2019m09{}.asf'.format(str(f))
    #print(video_name)
    video = cv2.VideoCapture(f)
    print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    count = originalCount = 0
    all_images = []
    imlist = []
    result = pd.DataFrame([])
    increment = 0
    while True:
        count += 1
        originalCount += 1
        ret,frame = video.read()
        if ret:
            all_images.append(frame)
            imlist.append('{}.jpg'.format(originalCount))
        else:
            result_ = Detect(all_images,imlist,model)
            result_ = list(result_)
            temp_df = pd.DataFrame(result_)
            temp_df.reset_index(drop=True)
            temp_df[0] = pd.to_numeric(temp_df[0])
            temp_df[0] = temp_df[0].apply(lambda x:x+ increment*BatchSize)
            all_images=[]
            imlist=[]
            count =0 
            increment += 1
            frames = [result,temp_df]
            result = pd.concat(frames) 
            break

        if count>=BatchSize:
            result_ = Detect(all_images,imlist,model)
            result_ =  list(result_)
            temp_df = pd.DataFrame(result_)
            temp_df.reset_index(drop=True)
            temp_df[0] = pd.to_numeric(temp_df[0])
            temp_df[0] = temp_df[0].apply(lambda x:x+ increment*BatchSize)
            all_images=[]
            imlist=[]
            count =0 
            increment += 1
            frames = [result,temp_df]
            result = pd.concat(frames)
    
    yolo_detections_fileName = f +"_{}_yolo.csv".format(originalCount)
    print(yolo_detections_fileName)
    result.to_csv(yolo_detections_fileName)
    gc.collect()
    all_images = []
    #call(['python3','applySort.py','--file',yolo_detections_fileName])
    os.system('python applySort.py --file '+ yolo_detections_fileName)
    
