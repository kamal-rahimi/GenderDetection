"""
"""

import time
from predict import prepare_image, indetify_gender, indetify_race, display_image
import cv2

def main():

    # Get a reference to webcam
    video_capture = cv2.VideoCapture("/dev/video0")

    time.sleep(2.0)

    while (True):
        ret, frame = video_capture.read()
        image, face = prepare_image(frame)
        predicted_gender , prob_gender = indetify_gender(face)
  
        display_image(image, predicted_gender, prob_gender, wait_time=1)


if __name__ == "__main__":
    main()