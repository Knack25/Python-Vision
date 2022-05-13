# Sample code from TechVidvan

# import necessary packages

from asyncio.subprocess import STDOUT
from multiprocessing.sharedctypes import Value
from re import sub
from tokenize import String
from time import sleep
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from py4j.java_gateway import JavaGateway


gateway = JavaGateway()
NetTranslate = gateway.entry_point.getNet()
angle = 0

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)


print("Starting Python Loop")
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''
    #print(p.stdout.readline())

    # post process the result
    if result.multi_hand_landmarks:
        lmxmin = 1028
        lmxmax = 0
        lmymin = 1028
        lmymax = 0
        landmarks = []
        xVals = []
        yVals = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                    
                landmarks.append([lmx, lmy])
                xVals.append(lmx)
                yVals.append(lmy)

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            
            lmxmin = min(xVals)
            lmxmax = max(xVals)
            lmymin = min(yVals)
            lmymax = max(yVals)
            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

            

        NetTranslate.setxEntry(lmx)
        NetTranslate.setxMinEntry(lmxmin)
        NetTranslate.setxMaxEntry(lmxmax)
        NetTranslate.setyEntry(lmy)
        NetTranslate.setyMinEntry(lmymin)
        NetTranslate.setyMaxEntry(lmymax)
        NetTranslate.setNameEntry(className)
        NetTranslate.setCameraAngle(angle)
            


    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()