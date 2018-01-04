import cv2
import sys

faceCascade = cv2.CascadeClassifier("haar_cascade/face.xml")

image = cv2.imread("a.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

print("Found {0} faces!". format(len(faces)))
#print (faces)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)


cv2.imshow("Faces found", image)
cv2.waitKey(0)