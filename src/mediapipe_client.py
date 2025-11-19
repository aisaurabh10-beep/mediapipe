import subprocess
import pickle
import struct
import os
import logging
import atexit

logger = logging.getLogger(__name__)

class MediaPipeClient:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.start_worker()
        atexit.register(self.stop)
        

    def start_worker(self):
        """Launches the MediaPipe worker using configured python.exe path."""
        try:
            python_exe_path = self.config.get('MediaPipe_Service', 'mp_python_exe_path')
        except Exception:
            logger.critical("FATAL: 'mp_python_exe_path' not set in config.ini")
            raise
            
        script_path = os.path.join('src', 'mediapipe_worker.py')
        
        if not os.path.exists(python_exe_path):
            logger.critical(f"FATAL: Python executable not found: {python_exe_path}")
            raise FileNotFoundError(python_exe_path)

        if not os.path.exists(script_path):
            logger.critical(f"FATAL: Worker script not found: {script_path}")
            raise FileNotFoundError(script_path)

        command = [python_exe_path, script_path]
        
        logger.info(f"Starting MediaPipe worker: {' '.join(command)}")
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info(f"MediaPipe worker started with PID: {self.process.pid}")
        except Exception as e:
            logger.critical(f"Failed to launch MediaPipe worker: {e}")
            raise

    def _log_worker_error(self):
        """Read and log stderr from crashed worker."""
        try:
            stderr_output = self.process.stderr.read().decode('utf-8')
            if stderr_output:
                logger.error("--- MediaPipe Worker Error ---")
                logger.error(stderr_output.strip())
                logger.error("-------------------------------")
        except Exception as e:
            logger.error(f"Failed to read worker stderr: {e}")

    def _write_message(self, data):
        """Write length-prefixed message to worker stdin."""
        try:
            len_bytes = struct.pack('!I', len(data))
            self.process.stdin.write(len_bytes)
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (IOError, BrokenPipeError, OSError) as e:
            logger.error(f"Failed to write to worker: {e}")
            self._log_worker_error()
            raise

    def _read_message(self):
        """Read length-prefixed message from worker stdout."""
        try:
            len_bytes = self.process.stdout.read(4)
            if not len_bytes:
                raise EOFError("Worker closed stdout")
            msg_len = struct.unpack('!I', len_bytes)[0]
            if msg_len == 0:
                raise EOFError("Worker sent zero-length message")
            msg_data = self.process.stdout.read(msg_len)
            if len(msg_data) < msg_len:
                raise EOFError("Incomplete message from worker")
            return msg_data
        except (IOError, struct.error, EOFError) as e:
            logger.error(f"Failed to read from worker: {e}")
            self._log_worker_error()
            raise

    def get_alignment_and_pose(self, face_crop_bgr):
        """
        Send face crop to worker, get back aligned face + pose.
        Returns: (aligned_face_bgr, yaw, pitch, roll, aspect_ratio)
        """
        try:
            if face_crop_bgr is None or face_crop_bgr.size == 0:
                logger.warning("Empty face crop")
                return None, 0.0, 0.0, 0.0, 0.0
                
            data = pickle.dumps(face_crop_bgr)
            self._write_message(data)
            
            response_data = self._read_message()
            if response_data is None:
                return None, 0.0, 0.0, 0.0, 0.0
                
            response = pickle.loads(response_data)
            
            # Check for error response
            if len(response) >= 5 and isinstance(response[-1], str):
                logger.warning(f"Worker error: {response[-1]}")
                return None, 0.0, 0.0, 0.0, 0.0
                
            if len(response) == 5:
                aligned_face, yaw, pitch, roll, aspect_ratio = response
                return aligned_face, yaw, pitch, roll, aspect_ratio
            else:
                # Fallback for old 4-tuple format
                aligned_face, yaw, pitch, roll = response
                return aligned_face, yaw, pitch, roll, 1.0

        except Exception as e:
            logger.error(f"MediaPipe communication error: {e}")
            raise

    def stop(self):
        if self.process:
            logger.info("Stopping MediaPipe worker...")
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("MediaPipe worker stopped")
            except (IOError, subprocess.TimeoutExpired, BrokenPipeError):
                self.process.kill()
                logger.warning("MediaPipe worker force-killed")