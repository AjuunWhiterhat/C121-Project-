import cv2 
import time 
import numpy as np
import keyboard
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))

cap = cv2.VideoCapture(0)

imgBg = cv2.imread("bangkok.jpg")
cap.set(3,640)
cap.set(4,480)

time.sleep(2)
image = 0

segmentor=SelfiSegmentation()

    


while (cap.isOpened()):

    ret,image = cap.read()
    image = np.flip(image,axis=1)
    image=segmentor.removeBG(image,imgBg,threshold=0.95)
    
    ret,frame = cap.read()

    if not ret:
        break
    frame = np.flip(frame,axis=1)
    
    frame = cv2.resize(frame,(640,480))
    image = cv2.resize(image,(640,480))

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    u_black = np.array([104,153,70])
    l_black = np.array([30,30,0])

    mask = cv2.inRange(frame,l_black,u_black)

    res = cv2.bitwise_and(frame,frame,mask=mask)

    f = frame - res
    f = np.where(f == 0, image, f)

    cv2.imshow("Real Video",f)
    cv2.imshow("Masked Video",image)

    cv2.waitKey(1)

    if keyboard.read_key() == "esc":
        break

    elif keyboard.read_key() == "q":
        break

cap.release()
out.release()
cv2.destroyAllWindows()