import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
 
from playsound import playsound

from EyeClosed.eye_closed import runEAR
from MouthClosed.mouth_closed import runMAR
from HeadPose.head_pose import getHeadPose
from PhoneDetection.phone_detection import phoneDetection

import os

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

import argparse

parser = argparse.ArgumentParser(description='Program for alertness detection during driving in traffic')

parser.add_argument('-c', '--camera', help='Camera index', default=0, type=int)
parser.add_argument('-e', '--ear', help='Eye Aspect Ratio Threshold', default=35)
parser.add_argument('-m', '--mar', help='Mouth Aspect Ratio Threshold', default=60)
parser.add_argument('--cfg', help='Pipeline config for object detection model', default='model\pipeline.config')
parser.add_argument('--ckpt', help='Checkpoint for model', default='model\checkpoint')
parser.add_argument('--labels', help='Label Map for model', default='model\mscoco_label_map.pbtxt')
parser.add_argument('--alarm', help='custom alarm mp3', default='beep.mp3')
parser.add_argument('--counter', help='Delay for alarm call', default=50, type=int)
args = parser.parse_args()

# Cell Phone detection paths

PATH_TO_CFG = args.cfg
PATH_TO_CKPT = args.ckpt
PATH_TO_LABELS = args.labels

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
# Mediapipe stuff for head pose and eye pos and mouth pose

cap = cv2.VideoCapture(args.camera)
detector = FaceMeshDetector(maxFaces=1, getZCoords=True)
 
counter = 0
color = (255, 0, 255)

def main():
    global counter, color
    while cap.isOpened():
        ret, img = cap.read()

        # MobileNet detection here
        image_np_with_detections, phone_flag = phoneDetection(img, detection_model, category_index)
        
        counter += 1

        image_np_with_detections, faces, originalVals = detector.findFaceMesh(image_np_with_detections)
        eye_flag = runEAR(image_np_with_detections, faces, detector)
        mouth_flag = runMAR(image_np_with_detections, faces, detector)
        head_pose_flag = getHeadPose(image_np_with_detections, faces, originalVals)

        if not eye_flag or mouth_flag or head_pose_flag or phone_flag:
            counter += 1
        else:
            counter = 0
        if faces:
            # If driver misbehaviour exists for more than stipulated frames, sound alarm
            if counter > args.counter:
                blinkCounter = 'Sound Alarm'
                playsound(args.alarm, block=False)
                cvzone.putTextRect(image_np_with_detections, f'{blinkCounter}', (50, 100),
                                colorR=color)

        # Show the final images
        cv2.imshow("Processed image", image_np_with_detections)
        cv2.imshow("Dashboard Camera",img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

