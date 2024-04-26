import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
#mediapipe was created by google and it can identify lots of stuff
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

mpHands = mp.solutions.hands
#an instance of mediapipe hands solutions
hands= mpHands.Hands() 
mpDraw = mp.solutions.drawing_utils
#used to draw the mediapipe recognitions on our image
xList = []
yList = []
lmList= []

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #CV2 works with BGR image but mediapipe only accepts RGB, hence convert
    results= hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
#if we just print, it will print landmarks , but we want to show them on the image, plot em
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
               # print(id, lm), prints ID and location of each landmark in the form of ratio of whole image, not the actual coordinates
               h, w, c = img.shape
               #print(h, w, c)  prints the height width and center of our input image
               cx, cy = int(lm.x*w), int(lm.y*h)
               xList.append(cx)
               yList.append(cy)
               lmList.append([id, cx, cy])
            
               #print(id, cx, cy)
            #    if id==8 or id == 4:
            #     cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)    
            #this above 2 lines were to circle the landmark numbers 4 and 8
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if len(lmList) !=0:
        print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img, (x1,y1), 15, (255,0,0), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,0), cv2.FILLED)


    cTime = time.time()
    fps= 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 3)
    #this is to print the FPS value on the image


    cv2.imshow("Image", img)
    
    # Check for 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
