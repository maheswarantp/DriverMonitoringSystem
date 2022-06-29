import cv2
import cvzone
import numpy as np

pts = [33, 263, 1, 61, 291, 199]

def getHeadPose(img, faces, originalVals):
    img_h, img_w, c = img.shape
    
    if originalVals:
        face_2d = []
        face_3d = []

        face = originalVals[0]
        # Get nose_2d and nose_3d
        nose_2d = (face[1][0] * img_w, face[1][1] * img_h)
        nose_3d = (face[1][0] * img_h, face[1][1] * img_h, face[1][2] * 8000)

        for i in pts:
            face_2d.append([int(face[i][0] * img_w), int(face[i][1] * img_h)])
            face_3d.append([int(face[i][0] * img_w), int(face[i][1] * img_h), face[i][2]])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w

        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360

        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
        cv2.line(img, p1, p2, (255, 0, 0), 2)

        if y  >= -8.5 and y <= 8.5 and x >= -10:
            return False
        return True