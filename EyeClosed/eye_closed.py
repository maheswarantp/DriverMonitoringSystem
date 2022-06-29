import cv2

idList_Left = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
idList_Right =[463, 359, 252, 253, 254, 256, 339, 384, 385, 386, 387, 388]

ratioList_Left = []
ratioList_Right = []

ratioList = []
color = (255, 0, 0)

def plot_eyes(img, face, idList):
    for id in idList:
        cv2.circle(img, face[id][:-1], 5, color, cv2.FILLED)
    
def getEyeCoords(img, face, eye_pos, detector, draw=False):
    
    up = face[eye_pos[0]][:-1]
    down = face[eye_pos[1]][:-1]
    left = face[eye_pos[2]][:-1]
    right = face[eye_pos[3]][:-1]

    lenghtVer, _ = detector.findDistance(up, down)
    lenghtHor, _ = detector.findDistance(left, right)

    if draw:
        cv2.line(img, up, down, (0, 200, 0), 3)
        cv2.line(img, left, right, (0, 200, 0), 3)
    
    return lenghtVer, lenghtHor

def getEAR(lengthVer, lengthHor):
    ratio = int((lengthVer / lengthHor) * 100)
    ratioList.append(ratio)
    if len(ratioList) > 3:
        ratioList.pop(0)
    ratioAvg = sum(ratioList) / len(ratioList)
    return ratioAvg

def runEAR(img, faces, detector, eye_EAR=35):
    if faces:
        face = faces[0]

        # Plot Eyes
        plot_eyes(img, face, idList_Left)
        plot_eyes(img, face, idList_Right)
    
        lengthVerLeft, lengthHorLeft = getEyeCoords(img, face, [159, 23, 130, 243], detector)
        lengthVerRight, lengthHorRight = getEyeCoords(img, face, [386, 253, 359, 463], detector)

        # left eye ear
        ratio_left = getEAR(lengthVerLeft, lengthHorLeft)
        ratio_right = getEAR(lengthVerRight, lengthHorRight)

        if ratio_left < eye_EAR and ratio_right < eye_EAR:
            return False
    return True