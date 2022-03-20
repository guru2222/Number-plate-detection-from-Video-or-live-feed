import cv2
import pytesseract
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
count = 0
#count = 1
cap = cv2.VideoCapture('video.MOV')

def processframes(frame):
    count= 0
    NumberPlates = cascade.detectMultiScale(gray, 1.2, 4)
    #print(type(NumberPlates))
    for(x, y, w, h) in NumberPlates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "number plate", (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        cropped = frame[y:y+h, x:x+w]
        cv2.imshow("number plate", cropped)
        count = count + 1
    cv2.imshow("Live Feed", frame)
    
while(cap.isOpened()):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processframes(gray)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
         
cap.release()
cv2.destroyAllWindows()