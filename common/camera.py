"""Source code: https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/common"""
import dataclasses
import pickle
from typing import Optional

import cv2
import numpy as np


@dataclasses.dataclass()
class Camera:
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    camera_matrix: np.ndarray = dataclasses.field(init=False)
    dist_coefficients: np.ndarray = dataclasses.field(init=False)
    camera_params_path: dataclasses.InitVar[str] = None

    def __post_init__(self, camera_params_path):
        with open(camera_params_path) as f:
            data = pickle.load(f)
        self.width = 640  # data['image_width']
        self.height = 480  # data['image_height']
        self.camera_matrix = np.array(data['mtx']).reshape(
            3, 3)
        self.dist_coefficients = np.array(
            data['dist']).reshape(-1, 1)

    def project_points(self,
                       points3d: np.ndarray,
                       rvec: Optional[np.ndarray] = None,
                       tvec: Optional[np.ndarray] = None) -> np.ndarray:
        assert points3d.shape[1] == 3
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float)
        if tvec is None:
            tvec = np.zeros(3, dtype=np.float)
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                        self.camera_matrix,
                                        self.dist_coefficients)
        return points2d.reshape(-1, 2)