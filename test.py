"""Test model"""
import argparse
import glob
import os

import urllib3
import ssl
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Net import LeNet5


url = 'http://10.194.55.211:8080/shot.jpg'
###you have to change the url

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)



def test(ckpt, gpu, webcam):
    """Run model testing"""

    model_info = torch.load(ckpt, map_location='cpu')
    model = LeNet5(ckpt=ckpt, gpu=gpu).eval()
    if gpu:
        model = model.cuda()
    
    transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if webcam:
        cap = cv2.VideoCapture(0)
    while True:
        
        ##################################################
        # Read data from the webcam
        ##################################################
        # Capture frame-by-frame
        if webcam:
            ret, frame = cap.read()
        else:
            http = urllib3.PoolManager()
            r = http.request('GET', url)
            # imgResp = urllib3.urlopen(url)
            imgNp = np.array(bytearray(r.data), dtype=np.uint8)
            frame = cv2.imdecode(imgNp, -1)

        height = frame.shape[0]
        width = frame.shape[1]
        if height>width:
            p = height - width
            p = int(p/2)
            frame = frame[p:p+width,:]
        if width>height:
            p = width - height
            p = int(p/2)
            frame = frame[:,p:p+height]
        showImage = frame
        # Our operations on the frame come here
        X = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
        X = transform(X)
        
        ##################################################
        # Pass the X here
        ##################################################
        X = X.unsqueeze(0)
        logit = model(X)
        logit = logit.squeeze()
        logit = torch.softmax(logit, dim=0)
        logit = logit.cpu().data.numpy()
        pred = np.argmax(logit)

        text = ""
        if pred==0:
            text = "Previous"
        elif pred==1:
            text = "Next"
        elif pred==2:
            text = "Stop"
        elif pred==3:
            text = "Other"
        ##################################################
        # Print the frame in the video
        ##################################################

        # draw the label into the frame
        __draw_label(frame, text, (20,20), (255,0,0))

        cv2.imshow('frame',showImage)
        k = cv2.waitKey(20) & 0xff
        if k == 27:
            break
    
    if webcam:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', 
                        help='Path to checkpoint file')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--webcam', action='store_true')

    args = parser.parse_args()
    test(args.ckpt, args.gpu, args.webcam)