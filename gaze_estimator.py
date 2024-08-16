import logging
from typing import List

import numpy as np

from common import Camera, Face
from common.face_model_mediapipe import FaceModelMediaPipe
from head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator

logger = logging.getLogger(__name__)



class GazeEstimator:
    # EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, ):
        # self._config = config
        self._face_model3d = FaceModelMediaPipe()
        self.camera = Camera("calib_cam0.yaml")
        self._normalized_camera = Camera("norm_cam.yaml")

        self._landmark_estimator = LandmarkEstimator()
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            1.0)
        # self._gaze_estimation_model = self._load_model()
        # self._transform = create_transform(config)

    # def _load_model(self) -> torch.nn.Module:
    #     model = create_model(self._config)
    #     checkpoint = torch.load(self._config.gaze_estimator.checkpoint,
    #                             map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     model.to(torch.device(self._config.device))
    #     model.eval()
    #     return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face)
        self._head_pose_normalizer.normalize(image, face)