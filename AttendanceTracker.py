#import libraries
import cv2
import numpy as np
import face_recognition
import os

#create list to get images from folder, generate encodings and try to find in webcam
path = 'Images'
images = []
classNames = []

#grab images from list
myList = os.listdir(path)
print(myList)

for cls in myList:
    #read current image
    currentImage = cv2.imread(f'{path}/{cls}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

#find encodings for each image(function)

def findEncodings(images):
    encodeList = []
    #loop through images
    for image in images:
        # Convert to BGR for OpenCV operations (since OpenCV uses BGR)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # Resize using OpenCV (which expects BGR)
        imgSmall = cv2.resize(image, (0, 0), fx=0.50, fy=0.50)
        encode = face_recognition.face_encodings(imgSmall)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnownFaces = findEncodings(images)
print(len(encodeListKnownFaces))

#find matches between encodings using the webcam

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    imageNew = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #Resize the image to reduce time but I already did that?

    faceLocationInCurrentFrame = face_recognition.face_locations(imageNew)
    encodeCurrentFrame = face_recognition.face_encodings(imageNew, faceLocationInCurrentFrame)

    #iterate through found faces and compare with known encodings
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceLocationInCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnownFaces, encodeFace)
        #find face distance
        faceDis = face_recognition.face_distance(encodeListKnownFaces, encodeFace)
        print(faceDis)
        #index of minimum value in our list(best match)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLocation
            #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0),2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (255, 0, 0),cv2.FILLED)
            cv2.putText(image, name, (x1 + 6, y2 -6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Video', image)
    cv2.waitKey(1)






