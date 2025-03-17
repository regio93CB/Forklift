"""
LINK YOLOV7 PER VEDERE IL CODICE ORIGINALE
https://github.com/WongKinYiu/yolov7/tree/main?tab=readme-ov-file


Per mandare lo script nella stazione definitiva, metterci echo per evitare di mettere la password:

echo babilonia | sudo -S python3 VBOM_NEW_TCP.py --weights runs/train/best_03.pt --conf 0.5 --source 0 --device 0

"""

import matplotlib
matplotlib.use('TkAgg')  # Sostituisci 'TkAgg' con il backend desiderato

import pickle, select 
import argparse, signal, sys, matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np
from numpy import random
import pandas as pd
import cv2, os, math
from openpyxl import load_workbook
import socket, struct
import subprocess as sp
#from cv2 import dnn_superres
import torch
import torch.backends.cudnn as cudnn
import time
import math
from termcolor import colored
import threading
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from pyzbar.pyzbar import decode, ZBarSymbol
import cv2

import traceback 
import datetime
from pathlib import PosixPath

#HW part
from multiprocessing import Process, set_start_method
import torch.multiprocessing as mp
import logging
import json

import can #sudo apt install python3-can
import logging

from customFunction import Simbolo, MyTimer, yolo2Matrix, vbom2Matrix, matrices_match, class_vbom2vbom, checkDevioAvailability, \
    packToSend, showWindows, getImages, getVBOMlist, setFocus, closeCam, calculate_time, checkLastResults, findResultsFromString, \
    fixture_check, right_webcam_check, takeCam, setExposure, checkCam, findButtonSymbol, takeFrontalFocus, takeExposure, writeExposure, \
    matrix_merge, define_two_max_exposure, saveFIXImage, pollingTera, new_matrices_match,\
    more_frequent,takeandsetAllExposure, checkAllignment, newcheckLastResults, to_scalar, setAllExposure, getCropValuesJson, \
    getCropValuesTxt, shutdown, reboot

def decode_QR(frame):
    x,y,w,h = 0,0,0,0
    text = ""
    carico = 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    barcodes = decode(thresh, symbols=[ZBarSymbol.QRCODE])

    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type  # dovrebbe essere "QRCODE"
        text = f"{barcode_data} ({barcode_type})"
        text2 = f"QR Code rilevato: {barcode_data} | Tipo: {barcode_type}"
        carico = 1

    return x,y,w,h,text,carico

def send_one(canCommandList):
    # a,b,c,d,e,f,g,h = canCommandList
    # Converti (arrotondando) i valori in interi
    byteValues = [int(round(x)) for x in canCommandList]
    for val in byteValues:
        if val > 255:
            raise ValueError(f"Il valore {val} non rientra in 8 bit.")

    message = bytes(byteValues)
    print(message)  # verrà stampata la sequenza di byte

    bus = can.interface.Bus(interface='pcan', channel='PCAN_USBBUS1', bitrate = 250000)
    msg = can.Message(
            arbitration_id=0x7FF, data=byteValues, is_extended_id=False
        )
    
    try:
        bus.send(msg)
        print(f"Message sent on {bus.channel_info}")
        bus.shutdown
    except can.CanError:
        print("Message NOT sent")
        bus.shutdown

class devioDetect:
    
    def __init__(self, parameters_file_index = 0):  #<<<<<<<<------------- RICORDA OGNI VOLTA DI USARE IL FOGLIO DI PARAMETRI GIUSTO
               
        #set camera resolution
        self.w, self.h, self.fps = 1280, 800, 30
        self.box_dimension_scale = 2.0
        self.camera = input("Inserisci il numero della telecamera: ")
        self.setupNN()

    #Setup the NN     
    def setupNN(self):
        
        print(colored("\n############### INIZIO SETUP RETE NEURALE ###############\n", "light_yellow"))        
        
        time1 = time.time()
        
        self.save_img = False 
        self.opt = self.add_all_argument()
        
        #Settaggio rete
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz, self.trace = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size, not self.opt.no_trace
        self.save_img = not self.opt.nosave and not self.source.endswith('.txt')  # save inference images
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        
        self.view_img  = True
        
        # Directories
        self.save_dir = Path(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        #Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.trace: self.model = TracedModel(self.model, self.device, self.opt.img_size)
        if self.half: self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        time3 = time.time()
        
        print(colored(f"\nTempo impiegato nel Setup della Rete Neurale: {time3 - time1:.0f} sec\n", "yellow"))
        print(colored("\n############### SETUP RETE NEURALE CONCLUSO CON SUCCESSO ###############\n", "yellow")) 
        
        time_i = time.time()
        
        #Apro le telecamere già settate
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.w, self.h, self.fps, 0, 0, self.w, self.h, None, None, None, None,  None, None, None, None, None, None, None, None, self.camera, img_size=self.imgsz, stride=self.stride)
        
        else: self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        
        time_f = time.time()
        print(colored(f"\nTempo impiegato per l'apertura della camera: {time_f - time_i:.0f} sec\n", "yellow"))
        
        print(colored(f"############### SETUP CAMERE CONCLUSO CON SUCCESSO ###############", "green"))
            
    


    
            
    #Main function
    def detection(self, save_img=False):
        
        for path, img, im0s, vid_cap in self.dataset:
                        
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3: img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.opt.augment)[0]

            # Inference
            with torch.no_grad(): pred = self.model(img, augment = self.opt.augment)[0]
                            
            #scelgo le classi, se le voglio tutte commento la seconda riga
            self.classes = self.opt.classes
            
            # Apply NMS
            pred, xfake = non_max_suppression(pred,  self.opt.conf_thres, self.opt.iou_thres, classes = self.classes, agnostic = self.opt.agnostic_nms)
                                            
            # Apply Classifier
            if self.classify: pred = apply_classifier(pred, self.model, img, im0s)
            
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                
                p = Path(p)  # to Path
                
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                
                if len(det):
                    
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    class_n_matrix = np.zeros((41, 2))
                    #time.sleep(0.666)

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                        class_n_matrix[int(c), 0] = int(c)
                        class_n_matrix[int(c), 1] = n

                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        
                        # print(f"C: {c}\nN: {n}\nS: {s}")
                    
                #Azzero la lista e i dari 
                coordinate_list, probs, total_prob = [], [], float(0)
                                    
                ########## QR CODE ##########
                print("-------------------------------")
                x,y,w,h,text, carico = decode_QR(im0)
                cv2.rectangle(im0, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(im0, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print(text)
                
                
                ########## QR CODE ##########
                                    
                #Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
                    if save_img or self.view_img:  # Add bbox to image
                        
                        #extract label from recognised class
                        label = self.names[int(cls)]
                        
                        #show with opencv the processed image                                                            !here you can manage the box dimension!
                        plot_one_box(xyxy, im0, label= f'{self.names[int(cls)]} {conf:.2f}', color=self.colors[int(cls)], line_thickness=int(self.box_dimension_scale))
                        
                        #convert the two point boxes in a center poit + size box
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        
                        #make a whole list of list 
                        line = [label, conf, xywh[0], xywh[1]]
                        coordinate_list.append(line)
                        
                        #Estraggo la probabilità media
                        for line in coordinate_list:
                            
                            prob = line[1]
                            probs.append(prob)
                                            
                for line in coordinate_list: 
                    
                    prob = float(line[1].item())
                    total_prob = total_prob + prob
                                
                    if probs: self.mean_confidence = sum(probs) / len(probs)

                #############RECEIVING CAN#############


                #################BOXES#################
                boxList = []
                valueList = []
                # print(coordinate_list)
                coord_size = 4
                boxN = 0
                wordN = 0
                for box in coordinate_list:
                    valueList = []
                    for word in box:
                        if wordN != 1:
                            # print(f" <{word}>")
                            # salva le coordinate
                            valueList.append(word)
                        wordN += 1
                    boxN += 1
                    if wordN == coord_size:
                        wordN = 0
                    boxList.append(valueList)
                             
                ##########BOXES###########
                
                #todo
                #se vedo 3 box allora TRACCIO I PUNTI MEDI
                avgPoint = []
                avgPointList = []
                print(boxList)
                
                if (boxN < 3):
                    print("\nnot enough boxes")
                elif (boxN > 3 ):
                    print("\ntoo many boxes")
                # ho riconosciuto il pallet e mando messaggio al canview4 che contiene info del QR e del baricentro delle box
                else:
                    print("\n3 boxes")
                    print(boxList)
                    #ordina boxList in base alla coordinata x
                    boxList.sort(key=lambda x: x[1])
                    print(boxList)
                    
                    #calcola il punto medio tra le box due a due
                    # avgPoint = [(boxList[0][2] + boxList[1][2]) / 2, (boxList[0][3] + boxList[1][3]) / 2]      
                                  
                    avgPoint = [(boxList[0][1] + boxList[1][1]) / 2, (boxList[0][2] + boxList[1][2]) / 2]  
                    avgPointList.append(avgPoint)
                    avgPoint = [(boxList[1][1] + boxList[2][1]) / 2, (boxList[1][2] + boxList[2][2]) / 2]         
                    avgPointList.append(avgPoint)
                    
                
                ##########BOXES###########

                ##########CAN SENDING###########


                # coppiaSinistra = da 0 a 30
                # coppiaDestra = da -30 a 0
                coppiaSinistra = 0
                coppiaDestra = 0
                centroSinistra = 0
                centroDestra = 0
                canCommand = [coppiaSinistra,coppiaSinistra, coppiaDestra,coppiaDestra,centroSinistra,centroDestra,carico,0]
                # #devo confrontare i punti medi ottenuti con le posizioni della forca, i quali ricevo via can
                
                send_one(canCommand)
                

                
                
                
                
                
                #intanto simulo l'output
                                        
                                        
                #     print("sending to HMI")
                #     #send to HMI
                #     #packToSend()
                    
                                             
                #################OUTPUT#################
                                    
                #Stream results
                if self.view_img:
                    
                    #Risistemo l0'immagine per renderla più piccola, almeno la vedo a modo
                    # im0 = cv2.resize((im0), (1280, 1040))
                            
                    cv2.imshow(str(p), im0)    
                
                    cv2.waitKey(1)  # 1 millisecond     
    
                  
    def add_all_argument(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='runs/train/best.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default= "0", help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.74, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.74, help='IOU threshold for NMS')
        parser.add_argument('--device', default="0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect1', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()
        print(opt)
        #check_requirements(exclude=('pycocotools', 'thop'))
        
        return opt

    
if __name__ == '__main__':

    try:
        D = devioDetect()  # Inizializza il tuo oggetto
        
        while True: 
            D.detection()
            
    except KeyboardInterrupt:
        print(colored("\nSistema interrotto Manualmente", "red"))

    except Exception as e:
        traceback.print_exc()
        
    finally:
        closeCam()