# Write a Python Script that captures images from your webcam video stream
# Extract all faces from the image frame (using haarcascades)
# Stores the face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
faceCascade = cv2.CascadeClassifier('../files/haarcascade_frontalface_default.xml')

skip = 0
faceData = []
datasetPath = '../data/'
fileName = input('Enter the name of person: ')

while True:
        ret, frame = cap.read()

        if ret == False:
                continue

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(frame, 1.3, 5)
        faces = sorted(faces, key = lambda f : f[2] * f[3], reverse = True)

        # Pick the largest face first
        for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Extract (Crop out the required face) : Region of Interest
                offset = 30
                faceSection = frame[y - offset : y + h + offset, x - offset : x + w + offset]
                faceSection = cv2.resize(faceSection, (100, 100))
                cv2.imshow('Face Section', faceSection)

                skip += 1
                if skip % 10 == 0:
                        faceData.append(faceSection)
        
        cv2.imshow('Frame', frame)

        keyPressed = cv2.waitKey(1) & 0xFF
        if keyPressed == ord('q'):
                break

# Convert our face list array into a numpy array
faceData = np.asarray(faceData)
faceData = faceData.reshape((faceData.shape[0], -1))
print(faceData.shape)

# Save this data into file system
np.save(datasetPath + fileName + '.npy', faceData)
print('Data successfully Saved!')

cap.release()
cv2.destroyAllWindows()

