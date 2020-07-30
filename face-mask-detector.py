# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import keyboard


count=0
def detect_and_predict_mask(frame, faceNet, maskNet):
	global count
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(x, y, end_coor_x, end_coor_y) = box.astype("int")
			(x, y) = (max(0, x), max(0, y))
			(end_coor_x, end_coor_y) = (min(w - 1, end_coor_x), min(h - 1, end_coor_y))
			face = frame[y:end_coor_y, x:end_coor_x]
			if len(face)!=0:
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = preprocess_input(face)
				faces.append(face)
				locs.append((x, y, end_coor_x, end_coor_y))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
url = 'http://192.168.43.1:8080/video'
vs = cv2.VideoCapture(url)
time.sleep(2.0)


# loop over the frames from the video stream
while True:
	rect, frame = vs.read()
	frame = imutils.resize(frame, width=480)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	for (box, pred) in zip(locs, preds):
		(x, y, end_coor_x, end_coor_y) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (x, y), (end_coor_x, end_coor_y), color, 2)
		cv2.rectangle(frame, (x,end_coor_y-30), (end_coor_x,end_coor_y), color, cv2.FILLED)
		cv2.putText(frame, label, (x+10,end_coor_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, 1)

		if label=='Mask':
			count+=1
		else:
			count-=1

		# if mask count is geater than 20 
		# input for the micro controller is 1 otherwise 0
		if count>=20:
			input_to_mc = 1
			count=0
			cv2.waitKey(10000) #delay to open and close the door
		else:
			input_to_mc = 0
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF

	# if the 'q' or 'ESC' key was pressed, break from the loop
	if keyboard.is_pressed('Esc') or key==ord('q'):
		print("[INFO] Program execution terminated by user")
		break

vs.release()
cv2.destroyAllWindows()