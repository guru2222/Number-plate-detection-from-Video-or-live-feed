import pytesseract
import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import Image
import cv2
import scipy.ndimage as ndimage

import numpy as np
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
#url='rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov'     
#site url:-- rtsp://admin:12345kota@192.168.1.253:554/cam/realmonitor?channel=1&subtype=0  
cap = cv2.VideoCapture('SimTrim.mp4')

def analyseframe(frame):
    NumberPlates = cascade.detectMultiScale(frame, 1.5, 6) #1.2,4
    #print(NumberPlates)
    if len(NumberPlates)==0:
        return False
    else:
        return True   
    
def processframes(frame):
    NumberPlates = cascade.detectMultiScale(frame, 1.5, 6)
    for(x, y, w, h) in NumberPlates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cropped = frame[y:y+h, x:x+w]
        cv2.imwrite('crop.jpeg',cropped)
        #print(type(cropped))
        img1 = np.array(Image.open('crop.jpeg'))
        cv2.imshow('window',img1) #cv2.imshow('name',file) python
        #img1 = cv2.bilateralFilter(img1, 11, 17, 17)
        text = pytesseract.image_to_string(img1)
        
        #REGEX
        #finalText = re.findall("[a-zA-Z]{2}[0-9]{2}[a-zA-Z]{2}[0-9]{4}", text)
        #if finalText==None:
            #flag=1
        #else:
            #flag=0   
        #print(text)
        #text=text.upper()
        
        print(text)
    cv2.imshow("Live Feed", frame)
    cv2.waitKey(1)
    
while(cap.isOpened()):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        trig = analyseframe(gray) 
        if trig:
            processframes(gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
         
cap.release()
cv2.destroyAllWindows()