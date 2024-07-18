import cv2 as cv
import datetime
face_cascade=cv.CascadeClassifier('C:/Users/ayushi yadav/OneDrive/Desktop/FOLDER/python/CV_projects/face_detection/haarcascade_frontalface_default.xml')
eye_cascade=cv.CascadeClassifier('C:/Users/ayushi yadav/OneDrive/Desktop/FOLDER/python/CV_projects/face_detection/haarcascade_eye_tree_eyeglasses.xml')
cap=cv.VideoCapture(0)

while cap.isOpened():
    ret,img=cap.read()
    data=str(datetime.datetime.now())
        # frame=cv2.putText(frame,text,(10,50),font,1,(255,255,255),2,cv2.LINE_AA)
    frame=cv.putText(img,data,(90,90),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2,cv.LINE_AA)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),5)
    cv.imshow('img',img)
    if cv.waitKey(1) &0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()