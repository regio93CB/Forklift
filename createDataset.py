import cv2 
from customFunction import checkCam, setFocus, setExposure, closeCam, getCropValuesJson, setAllExposure
import os 
from datetime import datetime
import json


class Parameters:
    def __init__(self, file_path="PARAMETERS.json"):
        self.file_path = file_path
        self.params = self.load_parameters()

    def load_parameters(self):
        with open(self.file_path, "r") as file:
            return json.load(file)

    def save_parameters(self):
        with open(self.file_path, "w") as file:
            json.dump(self.params, file, indent=4)

    def get(self, key, default=None):
        return self.params.get(key, default)

    def set(self, key, value):
        self.params[key] = value
        self.save_parameters()

parameters_file = "PARAMETERS.json"

#Inizializzo tutto 
cap0 = cap2 = cap4 = cap6 = None 
frame0 = frame2 = frame4 = frame6 = None
cropped_frame0 = cropped_frame2 = cropped_frame4 = cropped_frame6 = None

caps = []
parameters = Parameters(parameters_file).params

dataset_path = "/home/ae/VBOM/dataset/"
crop_vars = getCropValuesJson(parameters_file)

#Cerco le webcam disponibili
cameras = parameters["cameras"]

#Le mostro 
focus = parameters["focus"]
focus0 = focus["central"]    #Centrale
focus2 = focus["left"]       #sinistra
focus4 = focus["right"]      #Destra
focus6 = focus["frontal"]    #Frontale

exposure = parameters["exposure"]
exposure0 = exposure["central"]
exposure2 = exposure["left"]
exposure4 = exposure["right"]
exposure6 = exposure["frontal"]
        

h, w = 1920, 1080
if   int(len(cameras)) == int(1): fps = 30
elif int(len(cameras)) == int(2): fps = 15
elif int(len(cameras)) == int(3): fps = 10
elif int(len(cameras)) == int(4): fps = 5 

gst_str0 = (
            f'v4l2src device= /dev/video{0} ! '
            f'image/jpeg, width=(int){h}, height=(int){w}, framerate=(fraction){fps} ! '
            f'jpegparse ! jpegdec ! videoconvert ! '
            f'video/x-raw, format=(string)BGR ! appsink'
            )
gst_str2 = (
            f'v4l2src device= /dev/video{2} ! '
            f'image/jpeg, width=(int){h}, height=(int){w}, framerate=(fraction){fps} ! '
            f'jpegparse ! jpegdec ! videoconvert ! '
            f'video/x-raw, format=(string)BGR ! appsink'
            )
gts_str4 = (
            f'v4l2src device= /dev/video{4} ! '
            f'image/jpeg, width=(int){h}, height=(int){w}, framerate=(fraction){fps} ! '
            f'jpegparse ! jpegdec ! videoconvert ! '
            f'video/x-raw, format=(string)BGR ! appsink'
            )
gst_str6 = (
            f'v4l2src device= /dev/video{6} ! '
            f'image/jpeg, width=(int){h}, height=(int){w}, framerate=(fraction){fps} ! '
            f'jpegparse ! jpegdec ! videoconvert ! '
            f'video/x-raw, format=(string)BGR ! appsink'
            )

#Se ci sono webcam disponibili le inizializzo
if "0" in cameras: cap0 = cv2.VideoCapture(gst_str0, cv2.CAP_GSTREAMER)
if "2" in cameras: cap2 = cv2.VideoCapture(gst_str2, cv2.CAP_GSTREAMER)
if "4" in cameras: cap4 = cv2.VideoCapture(gts_str4, cv2.CAP_GSTREAMER)
if "6" in cameras: cap6 = cv2.VideoCapture(gst_str6, cv2.CAP_GSTREAMER)

#Inzializzo la lista delle webcam disponibili
for cap in cap0, cap2, cap4, cap6:
    
    if cap is not None: caps.append(cap)

if os.geteuid() == 0:

    import keyboard
    
    
    try:
        closeCam()
        setFocus(cameras, focus0, focus2, focus4, focus6)
        setAllExposure(exposure0, exposure2, exposure4, exposure6)

        while True:
            
            if cap0:
                
                ret0, frame0 = cap0.read()
                cropped_frame0 = frame0[crop_vars['crop_yi0']:crop_vars['crop_yf0'], crop_vars['crop_xi0']:crop_vars['crop_xf0']]
                cv2.imshow("Webcam0", cropped_frame0)
                
                
            if cap2:
                
                ret2, frame2 = cap2.read()
                cropped_frame2 = frame2[crop_vars['crop_yi2']:crop_vars['crop_yf2'], crop_vars['crop_xi2']:crop_vars['crop_xf2']]
                cv2.imshow("Webcam2", cropped_frame2)
                
            if cap4:
                
                ret4, frame4 = cap4.read()
                cropped_frame4 = frame4[crop_vars['crop_yi4']:crop_vars['crop_yf4'], crop_vars['crop_xi4']:crop_vars['crop_xf4']]
                cv2.imshow("Webcam4", cropped_frame4)
                
            if cap6:
                
                ret6, frame6 = cap6.read()
                cropped_frame6 = frame6[crop_vars['crop_yi6']:crop_vars['crop_yf6'], crop_vars['crop_xi6']:crop_vars['crop_xf6']]
                cv2.imshow("Webcam6", cropped_frame6)
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                    
            # Cambia devio se TIENI PREMUTO k
            if keyboard.is_pressed("s"):
                
                ora_corrente = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                ####0####   
                filename0 = f"Camera 0 - Ora: {ora_corrente}.JPG"
                cropped_filename0 = f"Cropped Camera 0 - Ora: {ora_corrente}.JPG"
                
                filepath = dataset_path + filename0
                cropped_filepath = dataset_path + cropped_filename0
                
                if frame0 is not None:           cv2.imwrite(filepath, frame0)
                if cropped_frame0 is not None:   cv2.imwrite(cropped_filepath, cropped_frame0)
                
                ####2####
                filename2 = f"Camera 2 - Ora: {ora_corrente}.JPG"
                cropped_filename2 = f"Cropped Camera 2 - Ora: {ora_corrente}.JPG"
                
                filepath = dataset_path + filename2
                cropped_filepath2 = dataset_path + cropped_filename2
                
                if frame2 is not None:          cv2.imwrite(filepath, frame2)
                if cropped_frame2 is not None:  cv2.imwrite(cropped_filepath2, cropped_frame2)
                
                ###4####
                filename4 = f"Camera 4 - Ora: {ora_corrente}.JPG"
                cropped_filename4 = f"Cropped Camera 4 - Ora: {ora_corrente}.JPG"
                
                filepath = dataset_path + filename4
                cropped_filepath4 = dataset_path + cropped_filename4
                
                if frame4 is not None:         cv2.imwrite(filepath, frame4)
                if cropped_frame4 is not None: cv2.imwrite(cropped_filepath4, cropped_frame4)

                
                ####6####
                filename6 = f"Camera 6 - Ora: {ora_corrente}.JPG"
                cropped_filename6 = f"Cropped Camera 6 - Ora: {ora_corrente}.JPG"
                
                filepath = dataset_path + filename6
                cropped_filepath6 = dataset_path + cropped_filename6
                
                if frame6 is not None:         cv2.imwrite(filepath, frame6)
                if cropped_frame6 is not None: cv2.imwrite(cropped_filepath6, cropped_frame6)

                print("Immagini salvate correttamente")
  
    except Exception as e:
        
        print("Errore: ", e)
        
    finally:
    
        closeCam()

        for cap in caps: cap.release()
        cv2.destroyAllWindows()
        
else:
    print("You need to run this script as root")
    exit()