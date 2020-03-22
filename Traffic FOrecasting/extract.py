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
    filename = arg.video
    f = filename.split('/')[1].split('.')[0]
    file1 = open("checkpoint/session.txt","r+")
    check = file1.read()
    if f in check.split(','):
        print('already processed, moving onto next')

    else:
        print("loading model")
        model = Darknet(arg.cfgfile)
        model.load_weights(arg.weightsfile)
        print("model loaded")
        BatchSize = 8 #frames
        lineOfInterest = [(10,250),(500,250)]

        #video_name = 'New export_r01_RUPALI-Sq-AP-RAM_MANDIR-OC-01(209)_y2019m09{}.asf'.format(str(f))
        #print(video_name)
        video = cv2.VideoCapture(filename)
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
                if (len(imlist) == len(all_images)) and len(imlist) != 0:
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
        file1.write(f + ',')
        file1.close()
        #call(['python3','applySort.py','--file',yolo_detections_fileName])
        os.system('python applySort.py --file {} --video {}'.format(yolo_detections_fileName,filename))


    
