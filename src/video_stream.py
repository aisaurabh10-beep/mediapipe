import cv2
import time
import logging
import threading

logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, config):
        self.rtsp_url = config.get('Camera', 'rtsp_url')
        try:
            # Try to convert to int for webcam
            self.rtsp_url = int(self.rtsp_url)
        except ValueError:
            pass # Keep as string for RTSP
            
        self.reconnect_delay = config.getfloat('Camera', 'reconnect_delay_seconds')
        
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _connect(self):
        logger.info(f"Connecting to video stream: {self.rtsp_url}...")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.error("Failed to open video stream.")
            self.cap = None
        else:
            logger.info("Video stream connected.")

    def _run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(self.reconnect_delay)
                self._connect()
                continue
            
            self.grabbed, self.frame = self.cap.read()
            
            if not self.grabbed:
                logger.warning("Stream disconnected. Reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_delay)

    def read(self):
        """Returns the latest frame."""
        return self.grabbed, self.frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap:
            self.cap.release()
        logger.info("Video stream stopped.")