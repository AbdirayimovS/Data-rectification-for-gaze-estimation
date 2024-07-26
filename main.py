# Adopted from https://github.com/Abecid/Data-Normalization-for-Appearance-Based-Gaze-Estimation

# references 
# 1. https://github.com/xucong-zhang/ETH-XGaze/blob/master/normalization_example.py
# 2. https://github.com/swook/faze_preprocess/blob/5c33caaa1bc271a8d6aad21837e334108f293683/create_hdf_files_for_faze.py
# 3. 3D Face Model is in https://github.com/xucong-zhang/ETH-XGaze/blob/master/face_model.txt
# 4. 3D Face Model is here: https://github.com/swook/faze_preprocess/blob/5c33caaa1bc271a8d6aad21837e334108f293683/sfm_face_coordinates.npy
# 4. MPIIGaze also provides the matlab file for the 3D Face Model 

import pickle

import numpy as np
import cv2
import mediapipe as mp
import scipy.io as sio



with open("calib_cam0.pkl", "rb") as file:
    calib_data = pickle.load(file)


# TEMPLATE_LANDMARK_INDEX = [
#     33, 160, 158, 144, 153, 133, # left eye indeces
#     263, 362, 380, 373, 384, 386, # left eye indeces from the man in the page point of view
#     168, 197, 5, 4, 2, # nose indeces
#     61, 291, # lips corner left-to-right
#     148, 152, 377 # chin
#     ]

# TEMPLATE_LANDMARK_INDEX = [1, 152, 33, 263, 61, 291] # Based on the 6-point based face model
TEMPLATE_LANDMARK_INDEX = [33, 133, 362, 263, 61, 291] # based on 2 right eye, 2 left eye and 2 mouth
# TEMPLATE_LANDMARK_INDEX = [
#     33, 160,  153, 133, # left eye indeces
#     263, 362,  384, 386, # left eye indeces from the man in the page point of view
#     168, 4, 2, # nose indeces
#     61, 291, # lips corner left-to-right
#     152 # chin
#     ]

TEMPLATE_FACE_3D_V1 = []
FACE_KEY_LANDMARK_INDEX = [0, 1, 2, 3, 4, 5] # [0, 9, 20]

camera_matrix = np.array(calib_data['mtx']) 
camera_distortion = np.array(calib_data['dist'])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Drawing const parameters
point_drawing = {'radius': 2, 'rgb': [0, 255, 0], 'thickness': 1}
point_center_drawing = {'radius': 5, 'rgb': [0, 0, 255], 'thickness': 2}

def draw_point(image, mat, spec):
    rgb = spec['rgb']
    for arr in mat:
        x = int(arr[0])
        y = int(arr[1])
        # Image is in BGR, flip the RGB order
        cv2.circle(image, (x, y), radius=spec['radius'], color=(rgb[2], rgb[1], rgb[0]), thickness=spec['thickness'])

def generate_3d_face():
    face = sio.loadmat('6_points-based_face_model.mat')['model']
    return face

def process_image(image, face_mesh):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return face_mesh.process(image)

def showInMovedWindow(winname, img, x, y):
    # Create a named window
    cv2.namedWindow(winname)
    # Move it to (x,y)
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname,img)

def estimateHeadPose(landmarks, face_model, camera, distortion, iteration=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    if iteration:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def normalizeFace(img, face, hr, ht, camera_matrix, center):
    focal_norm = 960
    distance_norm = 600
    roiSize=(300, 300)

    ht = ht.reshape((3, 1)) # head translation vector

    # Pose rotation vector converted to a rotation matrix
    hR = cv2.Rodrigues(hr)[0] # rotation matrix
    Fc = np.dot(hR, face) + ht # 3D positions of facial landmarks
    
    center = np.zeros(np.array(Fc[:, 0]).shape)
    for index in FACE_KEY_LANDMARK_INDEX:
        center += np.array(Fc[:,index])
    center = np.array([ center / len(FACE_KEY_LANDMARK_INDEX) ]).reshape((3,1))



    # actual distance bcenterween eye and original camera
    distance = np.linalg.norm(center) # actual distance between eye and original camera
    z_scale = distance_norm/distance

    # C_n: camera projection matrix for the normalized camera
    cam_norm = np.array([
       [focal_norm, 0, roiSize[0]/2],
       [0, focal_norm, roiSize[1]/2],
       [0, 0, 1.0],
    ])

    # scaling matrix
    S = np.array([ 
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, z_scale],
    ])
    
    # x_r: x-axis of the head coordinate system
    hRx = hR[:,0]

    # z-axis
    forward = (center/distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down) 
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)

    # rotation matrix R
    R = np.c_[right, down, forward].T # rotation matrix R
    
    # Transformation Matrix M (For 3D input)
    M = np.dot(S, R)

    # transformation matrix 
    W = np.dot(cam_norm, np.dot(M, np.linalg.inv(camera_matrix)))
        
    # image normalization
    img_warped = cv2.warpPerspective(img, W, roiSize)
    return img_warped

def face_detect(image_shape, multi_face_landmark):
    height = image_shape[0]
    width = image_shape[1]

    landmarks = np.empty((0,2), dtype=np.float64)
    for index in TEMPLATE_LANDMARK_INDEX:
        landmark = multi_face_landmark.landmark[index]
        landmarks = np.append(landmarks, np.array([[min(width, landmark.x*width), min(height, landmark.y*height)]]), axis=0)
    return landmarks, face_center(landmarks)

def face_center(landmarks):
    center = np.zeros(np.array(landmarks[0]).shape, dtype=np.float64)
    for index in FACE_KEY_LANDMARK_INDEX:
        center += np.array(landmarks[index])
    return np.array([ center / len(FACE_KEY_LANDMARK_INDEX) ])

def main():
    # Find 3D Standard Face Points
    face = generate_3d_face() # the template to compare the objects 
    num_pts = face.shape[1] # COMMENT: I add for the 6-point face model 
    facePts = face.reshape(num_pts, 1, 3)
   

    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()

            image = cv2.undistort(image, camera_matrix, camera_distortion) # I added it

            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            results = process_image(image, face_mesh)

            if results.multi_face_landmarks:
                # Get key face landmarks in 2D pixels
                landmarks, center = face_detect(image.shape, results.multi_face_landmarks[0])
                image.flags.writeable = True
                draw_point(image, landmarks, point_drawing) # the landmarks
                draw_point(image, center, point_center_drawing) # the green is the cente of the face
                
                # Convert 2D landmark pixels to 3D
                landmarks = landmarks.astype(np.float32)
                
                landmarks = landmarks.reshape(num_pts, 1, 2)

                # Get rotational/translation shift
                hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, camera_distortion) # solvePnP needs 3D model for comparison with landmarks of mediapipe
         
                processed_face2 = normalizeFace(image, face, hr, ht, camera_matrix, center) # COMMENT: for 6-point based model

                # Show camera image with landmarks
                cv2.imshow("Cam image", image)
                # Show normalized image
                cv2.imshow("normalized Image", processed_face2)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                print("Program is exited!")
                cap.release()
                cv2.destroyAllWindows()
                break
            if cv2.waitKey(5) & 0xFF == ord("s"):
                cv2.imwrite("demo_camera_image.png", image)
                cv2.imwrite("demo_final_data_rectificated_img.png", processed_face2)
                print("Demo images are saved!")


if __name__ == '__main__':
    main()



"""
Copyright 2019 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""