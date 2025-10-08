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



