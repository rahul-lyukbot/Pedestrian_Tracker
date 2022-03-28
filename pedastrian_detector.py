import cv2
from random import randrange as r
import numpy as np

# Getting haarcascades file for calcutlation and detection of pedestrian in frames
file = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
#Getting video for pedestrian data
video_for_read = cv2.VideoCapture("People_walk.mp4") ### if you want to get data from webcam then use this  **"video_for_read = cv2.VideoCapture(0)"**

# Using while loop for continous running the program til the break statement satisfy
while True:
    succesfull_frame,frame=video_for_read.read()
    # Converting the video into grayscale for better calculation
    gray_video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Now Detecting the coordinates of pedestrians in frames
    coordinates_of_frames = file.detectMultiScale(gray_video)
    # Now Adding rectangle arround the pedestrians
    for (x,y,w,h) in coordinates_of_frames:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(r(110),r(210),r(65)),4)
        
    
    
    cv2.imshow("Pedestrian_tracker",frame)
    key = cv2.waitKey(1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
   
video_for_read.release()   
cv2.destroyAllWindows()
