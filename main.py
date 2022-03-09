from typing import Tuple
import time
import cv2
import numpy as np

class Detector():
    def __init__(self) -> None:
        self.marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        return None

    def saveMarkerImage(self)->None:
        id_num = 0
        side_pixels = 100
        img = cv2.aruco.drawMarker(self.marker_dict, id_num, side_pixels)
        cv2.imwrite("marker_"+str(id_num)+".png", img)
        return None

    def detectMarkerInFrame(self, frame: np.ndarray)->Tuple:
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, self.marker_dict)
        return corners, ids, rejected_img_points

    def detectMarkerInStream(self, camera_stream: int = 2)->None:
        cap = cv2.VideoCapture(camera_stream)
        if not cap.isOpened():
            print("Failed to open capture device.")
            exit()
        stop = False
        while not stop:
            ret, frame = cap.read()
            corners, ids, rejected_img_points = self.detectMarkerInFrame(frame)
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if not ret:
                print("Frame not recieved.")
                stop = True
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) == ord('q'):
                stop = True

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = Detector()
    detector.detectMarkerInStream()
