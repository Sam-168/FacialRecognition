import cv2
import numpy as np
import face_recognition

# Import image - face_recognition loads as RGB
imgSam = face_recognition.load_image_file("Images/myPic.png")

# Convert to BGR for OpenCV operations (since OpenCV uses BGR)
imgSamBGR = cv2.cvtColor(imgSam, cv2.COLOR_RGB2BGR)

# Resize using OpenCV (which expects BGR)
imgSamSmall = cv2.resize(imgSamBGR, (0, 0), fx=0.50, fy=0.50)

# Convert back to RGB for face_recognition
imgSamRGB = cv2.cvtColor(imgSamSmall, cv2.COLOR_BGR2RGB)

# Import imageTest
imgTest = face_recognition.load_image_file("Images/reference.jpg")
# Convert to BGR for OpenCV
imgTestBGR = cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR)
# Resize imgtest
imgTestSmall = cv2.resize(imgTestBGR, (0, 0), fx=0.15, fy=0.15)
# Convert back to RGB for face_recognition
imgTestRGB = cv2.cvtColor(imgTestSmall, cv2.COLOR_BGR2RGB)


# Get face location - use the RGB converted image
faceLocation = face_recognition.face_locations(imgSamSmall)[0] #Single image = get first element
encodeSam = face_recognition.face_encodings(imgSamSmall)[0]
#draw a rectangle for top, right, bottom, left co-ordinates on face
cv2.rectangle(imgSamSmall, (faceLocation[3], faceLocation[0], faceLocation[1], faceLocation[2]), (255, 0, 255), 2)

cv2.imshow("Sam Ndlela", imgSamSmall)
cv2.imshow("Test Ndlela", imgTestRGB)
cv2.waitKey(0)
