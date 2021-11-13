# Import required modules
from face_aligner.FaceAligner import FaceAligner
import numpy as np
import dlib
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class FaceOperator:
    FACE_PROTO_FILENAME = "opencv_face_detector.pbtxt"
    FACE_MODEL_FILENAME = "opencv_face_detector_uint8.pb"
    SHAPE_PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"

    def __init__(self, models_dir="models", box_padding=20, desiredFaceWidth=224, desiredFaceHeight=224):
        self.padding = box_padding
        faceProto = os.path.join(models_dir, self.FACE_PROTO_FILENAME)
        faceModel = os.path.join(models_dir, self.FACE_MODEL_FILENAME)
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(
            models_dir, self.SHAPE_PREDICTOR_FILENAME))
        self.fa = FaceAligner(
            self.predictor, desiredFaceWidth=desiredFaceWidth, desiredFaceHeight=desiredFaceHeight)

    def _pad_bb(self, rect, shape):
        # Add padding to the bbox taking into account the image shape
        rect[0] = max(0, rect[0]-self.padding)
        rect[1] = max(0, rect[1]-self.padding)
        rect[2] = min(rect[2]+self.padding, shape[1]-1)
        rect[3] = min(rect[3]+self.padding, shape[0]-1)

        return rect

    def _getFaceBox(self, frame):
        frameOpencv2Dnn = frame.copy()
        frameHeight = frameOpencv2Dnn.shape[0]
        frameWidth = frameOpencv2Dnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencv2Dnn, 1.0, (300, 300), [
                                     104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        bbox = None
        max_confidence = 0.0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max_confidence:
                max_confidence = confidence
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bbox = [x1, y1, x2, y2]
        return bbox

    def find_and_align(self, img):
        bbox = self._getFaceBox(img)
        bbox = self._pad_bb(bbox, img.shape)
        dlibRect = dlib.rectangle(int(bbox[0]), int(
            bbox[1]), int(bbox[2]), int(bbox[3]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aligned = self.fa.align(img, gray, dlibRect)
        return aligned

    def _get_landmarks(self, gray):
        faces = self.detector(gray, 1)
        landmarks = []
        if len(faces) == 0:
            return landmarks
        rect = faces[0]
        shape = self.predictor(gray, rect)
        for i in range(0, 68):
            p = (shape.part(i).x, shape.part(i).y)
            landmarks.append(p)
        return landmarks

    def _draw_landmarks(self, gray, landmarks=None):
        if landmarks == None:
            landmarks = self._get_landmarks(gray)
        c = (0, 0, 0)
        face = gray.copy()
        for p in landmarks:
            cv2.circle(face, p, 2, c, -1)
        return face

    def calculate_landmarks_distances(self, gray, landmarks=None, index=None, outliers=None):
        if landmarks == None:
            landmarks = self._get_landmarks(gray)
        if len(landmarks) == 0:
            if index is not None and outliers is not None:
                outliers.append(index)
            return np.zeros(67*68//2)
        distances = []
        for i, lm in enumerate(landmarks[:-1]):
            for o_lm in landmarks[i+1:]:
                distances.append(np.linalg.norm(np.array(lm)-np.array(o_lm)))
        return np.stack(distances, axis=0)
