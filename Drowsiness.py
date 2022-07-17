import os
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	EAR = (A + B) / (2.0 * C)
	return EAR
	
ear_threshold = 0.225
limit_for_drowsiness = 15
detecting_the_face_landmarks = dlib.get_frontal_face_detector()
predicting_the_location_of_eyes = dlib.shape_predictor("pretrained_face_detector.dat")

(one_end_of_left_eye, other_end_of_left_eye) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(one_end_of_right_eye, other_end_of_right_eye) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
camera_window=cv2.VideoCapture(0)
closed_eye_counter=0
while True:
	ret, frame=camera_window.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detecting_the_face_landmarks(gray, 0)
	for subject in subjects:
		shape = predicting_the_location_of_eyes(gray, subject)
		shape = face_utils.shape_to_np(shape)
		left_eye = shape[one_end_of_left_eye:other_end_of_left_eye]
		right_eye = shape[one_end_of_right_eye:other_end_of_right_eye]
		leftEAR = eye_aspect_ratio(left_eye)
		rightEAR = eye_aspect_ratio(right_eye)
		average_EAR = (leftEAR + rightEAR) / 2.0
		if average_EAR < ear_threshold:
			closed_eye_counter += 1
			if closed_eye_counter >= limit_for_drowsiness:
				os.system('say "Drowsy"')
		else:
			closed_eye_counter = 0
	cv2.imshow("",frame)
	key = cv2.waitKey(1)
	if key == ord("q"):
		break
cv2.destroyAllWindows()
camera_window.release()