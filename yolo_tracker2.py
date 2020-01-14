from models import *
from utils.utils import *
from utils.datasets import *

import torch
import torchvision.transforms as transforms

import cv2
import sys
import argparse

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from time import sleep


class YOLO():
    def __init__(self, opt):        
        self.opt = opt  
        self.img_size = opt.img_size      
        self.model = Darknet(self.opt.model_def, self.img_size)     # yolov3-tiny.cfg / 416          
        device = True if torch.cuda.is_available() else False        
        if device:
            self.model = self.model.to('cuda')
        if opt.weights_path.endswith(".weights"):
            self.model.load_darknet_weights(self.opt.weights_path)
        else:
            self.model.load_state_dict(torch.load(self.opt.weights_path))
        self.model.eval()
        classes = load_classes(self.opt.class_path)
        classes = load_classes(self.opt.class_path)                
        self.is_cuda = device        
        print('YOLO model initializing...', 'cuda=', device) 

    def find(self, frame):
        self.input_img = Image.fromarray(frame)
        trans=transforms.Compose([transforms.ToTensor()])
        self.input_img=trans(self.input_img).unsqueeze(0)
        if self.is_cuda:
            self.input_img = self.input_img.to('cuda')
        detections = self.model.forward(self.input_img)
        detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)[0]
    
        if detections is not None:
            for det in detections:
                if det[-1] != 0:
                    continue

                det=det.tolist()
                det[2]-=det[0]
                det[3]-=det[1]

                bbox=tuple(det[:4])
                return bbox

      
    def resize_numpy(self, img):
        h, w, c = img.shape
        if w > h :
            pad1 = (w - h) // 2         
            img = img[0:h, pad1:h+pad1, :]
        else:
            pad1 = (h - w) // 2     
            img = img[pad1:w+pad1, 0:w, :]           
        img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
        return img


class TrackingPlayer():
    def __init__ (self, source, windowName, opt):
        try :
            self.Yolo = YOLO(opt)
            self.video = cv2.VideoCapture(source)
            self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps  = self.video.get(cv2.CAP_PROP_FPS)
            self.wait_time = int(1000/self.fps) - 15
            self.windowName = windowName
            cv2.namedWindow(windowName)             
            #print( 'device={} opt.img_size={}'.format(device, opt.img_size ))                             
        except Exception as err:
            print('Err(init) :' , err)
       
 
    def _close(self):
        self.video.release()
        cv2.destroyAllWindows()
        exit()

    def setBBox(self):
        loop = True
        while (loop):
            ok, first_frame = self.video.read()                   
            resize_frame = self.Yolo.resize_numpy(first_frame)               
            resize_bbox  = self.Yolo.find(resize_frame)
            cv2.imshow(self.windowName, first_frame)
            if resize_bbox is not None:
                loop = False     
        self.bbox = rescale_boxes(resize_bbox, opt.img_size, ( self.video.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video.get(cv2.CAP_PROP_FRAME_WIDTH)))

    
    def play(self) :                
        try :
            self.setBBox()
            while(True):
                ret, frame = self.video.read()            
                resize_frame = self.Yolo.resize_numpy(frame)         
                resize_bbox  = self.Yolo.find(resize_frame)
                self.bbox    = rescale_boxes(resize_bbox, opt.img_size, ( self.video.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video.get(cv2.CAP_PROP_FRAME_WIDTH)))
                (x,y,w,h)=[int(v) for v in self.bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2, 1)                 
                cv2.imshow(self.windowName, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as err:
            print('Err(play) :' , err)
        finally:
            print('finally')
            self._close()
        


if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument("--model_def",        type=str,   default="config/yolov3.cfg",      help="path to model definition file")
    parser.add_argument("--weights_path",     type=str,   default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path",       type=str,   default="data/coco.names",        help="path to class label file")
    parser.add_argument("--conf_thres",       type=float, default=0.25,                     help="object confidence threshold")
    parser.add_argument("--nms_thres",        type=float, default=0.4,                      help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size",         type=int,   default=416,                      help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str,                                     help="path to checkpoint model")
    parser.add_argument("--video",            type=str,   default='video (6).avi',          help="path of video 1,4,6,10")
    opt=parser.parse_args()
    
    video_path=0
    if opt.video is not '':
        video_path=opt.video

    player = TrackingPlayer(video_path, "Main", opt)    
    player.play()

 
