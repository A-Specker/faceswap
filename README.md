# Facedetection Experiments

This repo is for testing the basics of faceswap.
## 1. Facedetection  
There are three different ways to detect the face:
- Haar Detector
- Histogram of Oriented Gradients
- Neural Network

Somehow, all seem to work good...

## 2. Facial Landmarks
The pose of the face is estimated by fitting 68 facial landmarks to significant points in the face.
From those points the face can be divided in triangles.

## 3. Swap Faces
Each triangle is then warped to the position of the corresponding triangle in the other face and smoothed a bit.
 
