"""
Code from 
https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/head_pose_estimation/face_landmark_estimator.py
Modified for my usage.
"""
from typing import List

import mediapipe
import numpy as np

from common import Face


class LandmarkEstimator:
    """LandmarkEstimator encapsulates the Mediapipe FaceMesh module."""
    def __init__(self):
        self.detector = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=1)

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """public method for detection faces"""
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
        return detected
