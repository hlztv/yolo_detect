from models import *
from utils.utils import *
from utils.datasets import *
from timeit import default_timer as timer

import cv2
import torch
import torchvision.transforms as transforms

import sys
import argparse
import os

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

class YOLO():
    def __init__ (self, opt, is_cuda, img_size=416):
        self.is_cuda = is_cuda
        self.opt = opt
        self.image_size = img_size
        #self.boxes, self.scores, self.classes = self.generate()
        self.model = Darknet(opt.model_def, opt.img_size)     # yolov3-tiny.cfg / 416          
        #for no, lt in enumerate(model.module_list):
        #   print(no, lt)   
        if is_cuda:
            self.model = self.model.to('cuda')
        if opt.weights_path.endswith(".weights"):
            self.model.load_darknet_weights(opt.weights_path)         # Load darknet weights
        else:
            self.model.load_state_dict(torch.load(opt.weights_path))     # Load checkpoint weights

        self.model.eval()   # Set in evaluation mode
        #classes = load_classes(opt.class_path)   # Extracts class labels from file

    def setSize(self, img_size):
        self.image_size = img_size

    def rebuild_boxes(self, boxes, rsize, original_shape):  
        orig_w, orig_h = original_shape                  
        if orig_w > orig_h:
            pad = (orig_w-orig_h) // 2
            dim = orig_h/rsize
            x = int (boxes[0]*dim + pad + 0.5)
            y = int (boxes[1]*dim + 0.5)
            w = int (boxes[2]*dim + 0.5)
            h = int (boxes[3]*dim + 0.5)
        else:    
            pad = (orig_h-orig_w) // 2
            dim = orig_w/rsize
            x = int (boxes[0]*dim + 0.5)
            y = int (boxes[1]*dim + pad + 0.5)
            w = int (boxes[2]*dim + 0.5)
            h = int (boxes[3]*dim + 0.5)
        #print('orig_w=', orig_w, 'orig_h=', orig_h)    
        return (x, y, w, h)  

    def detect_object(self, frame):
        start = timer()
        input_img = Image.fromarray(frame)
        trans     = transforms.Compose([transforms.ToTensor()])
        input_img = trans(input_img).unsqueeze(0)
        if self.is_cuda:
            input_img = input_img.to('cuda')

        # Detect Object
        with torch.no_grad():
            detections = self.model.forward(input_img)                              
            detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)

        if detections is not None:
            for det in detections:  
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in det:                                 
                    if int(cls_pred) != 0:
                        continue                                    
                    #print("\t+ x:%4d y:%4d w:%4d h:%4d Conf: %.5f Label: %s" % (x1, y1, w, h,  cls_conf.item(), classes[int(cls_pred)]))
                    out_boxes = ( x1, y1, x2-x1, y2-y1 )
                    boxes     = self.rebuild_boxes(out_boxes, self.opt.img_size, self.image_size)    
                    end       = timer()
                    #print(end - start)
                    return boxes, cls_conf.item()
        end = timer()
        return (0,0,0,0), 0.0       


def resize_image(img, img_size):
    h, w, c = img.shape    
    if w > h :
        pad1 = (w - h + 1 ) // 2        # 반올림
        img = img[0:h, pad1:h+pad1, :]
    else:
        pad1 = (h - w + 1) // 2     
        img = img[pad1:w+pad1, 0:w, :]           
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)    
    return img  

def detect_video(opt, video_path, output_path=""):    
    is_cuda = True if torch.cuda.is_available() else False
    fname_list = []    
    for root, dirs, files in os.walk(video_path):        
        rootpath = os.path.abspath(root)
        for fname in files:
            if fname.endswith('.mp4') or fname.endswith('.avi') :
                fname_list.append(os.path.join(rootpath, fname)) 

    for video_path in fname_list:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps    = vid.get(cv2.CAP_PROP_FPS)
        video_size   = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        isOutput = True if output_path !="" else False
        
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
            
        fps = "FPS:  ??"
        is_cuda = True if torch.cuda.is_available() else False
        print('YOLO (', video_path, ') cuda=', is_cuda)     
        yolo = YOLO(opt, is_cuda, video_size)

        accu_time = 0
        curr_fps = 0
        prev_time = timer()        
        while True:
            return_value, frame = vid.read()
            if return_value is False:
                break
            image_m   = resize_image(frame, opt.img_size)
            bboxs, conf = yolo.detect_object(image_m)         
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accu_time = accu_time + exec_time
            curr_fps  = curr_fps + 1
            if accu_time > 1:
                accu_time = accu_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            (x, y, w, h) = bboxs        
            cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2 )
            cv2.namedWindow("video", cv2.WINDOW_NORMAL)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2, 1) 
            cv2.imshow("video", frame)
            if isOutput:
                out.write(frame)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break    
        
        vid.release()
    cv2.destroyAllWindows()

def detect_image(opt, image_path):    
    is_cuda = True if torch.cuda.is_available() else False
    fname_list = []    
    for root, dirs, files in os.walk(image_path):        
        rootpath = os.path.abspath(root)
        for fname in files:
            if fname.endswith('.jpg') or fname.endswith('.png') :
                fname_list.append(os.path.join(rootpath, fname))            

    
    yolo = YOLO(opt, is_cuda)
    for fname_path in fname_list:
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)        
        image = cv2.imread(fname_path, cv2.IMREAD_ANYCOLOR )
        height, width, channel = image.shape        
        yolo.setSize((width, height))
        image_m   = resize_image(image, opt.img_size)
        bboxs, conf = yolo.detect_object(image_m)            
        (x, y, w, h) = bboxs                    
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2, 1) 
        cv2.imshow("image", image)   
        cv2.waitKey(0)
                        
    cv2.destroyAllWindows()    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_def",        type=str,   default="config/yolov3.cfg",      help="path to model definition file")
    parser.add_argument("--weights_path",     type=str,   default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path",       type=str,   default="data/coco.names",        help="path to class label file")
    parser.add_argument("--conf_thres",       type=float, default=0.25,                     help="object confidence threshold")
    parser.add_argument("--nms_thres",        type=float, default=0.4,                      help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size",         type=int,   default=416,                      help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str,                                     help="path to checkpoint model")
    parser.add_argument("--image",            type=str,   default="data/images",            help="path of images")
    parser.add_argument("--video",            type=str,   default="data/video",             help="path of video")
    #parser.add_argument("--video",           type=str,   default='Chaplin.mp4',            help="path of video 1,4,6,10")
    opt=parser.parse_args()    
    video_path = image_path = 0
    if opt.video is not '':
        video_path=opt.video
    if opt.image is not '':
        image_path=opt.image

    detect_video( opt, video_path )
    #detect_image( opt, image_path )
    
   