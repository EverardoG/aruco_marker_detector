"""Class for simple aruco marker detection"""
import atexit
import threading
from typing import Tuple
import time
import threading
import atexit
import logging

import cv2
import numpy as np

class ArucoMarkerDetector():
    def __init__(self, camera_stream: int, visualize_results: bool = True) -> None:
        # Initialize dictionary of aruco markers for easy reference
        self.marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        # Start the video capture
        self.cap = cv2.VideoCapture(camera_stream)
        if not self.cap.isOpened():
            logging.error("Failed to open capture device.")
        # Run seperate camera thread to continuously read frames from video capture
        self.failed_to_grab_cap = False
        self.camera_thread = threading.Thread(target=self._reader)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        # For visualizing detected marker
        self.visualize_results = visualize_results
        # Cleanup visualization
        atexit.register(self.cleanup)

    def _reader(self)->None:
        """Continuously read video capture.

        This works around the fact that the video capture gives us the NEXT
        frame instead of the LATEST frame. This process contiuously reads
        frames so that when we actually try to get a frame for marker detection,
        we get the LATEST frame."""
        while not self.failed_to_grab_cap:
            ret = self.cap.grab()
            if not ret:
                logging.error("_reader() failed to grab video capture.")
                self.failed_to_grab_cap = True
        return None

    def readFrame(self)->np.ndarray:
        """Reads latest frame from camera stream."""
        _, frame = self.cap.retrieve()
        return frame

    def detectCornersInFrame(self, frame: np.ndarray)->Tuple:
        """Detect the relevant aruco marker corners in the given frame."""
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, self.marker_dict)
        return corners, ids, rejected_img_points

    def detectMarker(self)->Tuple:
        """Detect the marker from the camera.
        Return the marker pose relative to the camera and corners in the image."""
        frame = self.readFrame()
        corners, ids, rejected_img_points = self.detectCornersInFrame(frame)
        if self.visualize_results:
            cv2.imshow("Aruco Marker Detector", frame)
        return corners, ids, rejected_img_points

    def cleanup(self)->None:
        self.cap.release()
        cv2.destroyAllWindows()
        return None

def detectMarkerInStream(camera_stream: int = 2)->None:
    cap = cv2.VideoCapture(camera_stream)
    if not cap.isOpened():
        print("Failed to open capture device.")
        exit()
    stop = False
    while not stop:
        ret = cap.grab()
        ret, frame = cap.retrieve()
        # corners, ids, rejected_img_points = self.detectMarkerInFrame(frame)
        # frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if not ret:
            print("Frame not recieved.")
            stop = True
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            stop = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # detectMarkerInStream()
    detector = ArucoMarkerDetector(2, True)
    time.sleep(3)
    # detector.detectMarker()
