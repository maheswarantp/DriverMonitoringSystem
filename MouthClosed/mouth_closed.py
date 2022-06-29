import cv2

idList = [61, 291, 39, 181, 0, 17, 269, 405]
ratioList = []
color = (255, 0, 0)

def plot_mouth(img, face, idList):
    for id in idList:
        cv2.circle(img, face[id][:-1], 5, color, cv2.FILLED)
    
def getMouthCoords(img, face, eye_pos, detector, draw=False):
    
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

def getMAR(lengthVer, lengthHor):
    ratio = int((lengthVer / lengthHor) * 100)
    ratioList.append(ratio)
    if len(ratioList) > 3:
        ratioList.pop(0)
    ratioAvg = sum(ratioList) / len(ratioList)
    return ratioAvg

def runMAR(img, faces, detector, MOUTH_EAR=60):
    if faces:
        face = faces[0]

        plot_mouth(img, face, idList)
    
        # lr 61, 291
        # up 0 17
        lengthVer, lengthHor = getMouthCoords(img, face, [0, 17, 61, 291], detector)

        ratio = getMAR(lengthVer, lengthHor)
        if ratio < MOUTH_EAR:
            return False
    return True