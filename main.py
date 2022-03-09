"""Class for simple aruco marker detection"""
from typing import Tuple
import time
import atexit

import cv2
import numpy as np

from bufferless_video_capture import BufferlessVideoCapture

class ArucoMarkerDetector():
    def __init__(self, camera_stream: int, visualize_results: bool = True) -> None:
        # Initialize dictionary of aruco markers for easy reference
        self.marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        # Start the video capture
        self.cap = BufferlessVideoCapture(camera_stream)
        # For visualizing detected marker
        self.visualize_results = visualize_results
        # Cleanup visualization
        atexit.register(self.cleanup)

    def detectCornersInFrame(self, frame: np.ndarray)->Tuple:
        """Detect the relevant aruco marker corners in the given frame."""
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, self.marker_dict)
        return corners, ids, rejected_img_points

    def detectMarker(self)->Tuple:
        """Detect the marker from the camera.
        Return the marker pose relative to the camera and corners in the image."""
        frame = self.cap.read()
        corners, ids, rejected_img_points = self.detectCornersInFrame(frame)

        if self.visualize_results:
            result_frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow("Aruco Marker Detector", result_frame)
            cv2.waitKey(1)

        return corners, ids, rejected_img_points

    def cleanup(self)->None:
        cv2.destroyAllWindows()
        return None

if __name__ == "__main__":
    # detectMarkerInStream()
    detector = ArucoMarkerDetector(2, True)
    while True:
        detector.detectMarker()
        time.sleep(0.5)

