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
        atexit.register(self.stop) # Ensure it's cleaned up on exit

    def start_worker(self):
        """
        Launches the worker script using the direct python.exe path.
        """
        # --- THIS IS THE NEW, ROBUST LAUNCH METHOD ---
        try:
            python_exe_path = self.config.get('MediaPipe_Service', 'mp_python_exe_path')
        except Exception:
            logger.critical("FATAL: 'mp_python_exe_path' not set in config.ini [MediaPipe_Service]")
            logger.critical("Please find your 'mp_env' python.exe path and add it to the config.")
            raise
            
        script_path = os.path.join('src', 'mediapipe_worker.py')
        
        # Check if the python.exe exists
        if not os.path.exists(python_exe_path):
            logger.critical(f"FATAL: Python executable not found at: {python_exe_path}")
            logger.critical("Please check the 'mp_python_exe_path' in your config.ini.")
            raise FileNotFoundError(python_exe_path)

        # Check if the worker script exists
        if not os.path.exists(script_path):
            logger.critical(f"FATAL: MediaPipe worker script not found at: {script_path}")
            raise FileNotFoundError(script_path)

        command = [python_exe_path, script_path]
        
        # --- END OF NEW LAUNCH METHOD ---
        
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
            logger.critical(f"Failed to launch MediaPipe worker with command: {' '.join(command)}")
            logger.critical(f"Error: {e}", exc_info=True)
            raise

    def _log_worker_error(self):
        """Helper function to read and log stderr from the crashed worker."""
        try:
            # Read any remaining error output
            stderr_output = self.process.stderr.read().decode('utf-8')
            if stderr_output:
                logger.error("--- MediaPipe Worker Traceback ---")
                logger.error(stderr_output.strip())
                logger.error("------------------------------------")
            else:
                logger.error("MediaPipe worker crashed with no stderr output.")
        except Exception as log_e:
            logger.error(f"Failed to read stderr from crashed worker: {log_e}")

    def _write_message(self, data):
        """Writes a length-prefixed message to the worker's stdin."""
        try:
            len_bytes = struct.pack('!I', len(data))
            self.process.stdin.write(len_bytes)
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (IOError, BrokenPipeError, OSError) as e:
            logger.error(f"Failed to write to MediaPipe worker (pipe broken): {e}")
            self._log_worker_error()
            raise e

    def _read_message(self):
        """Reads a length-prefixed message from the worker's stdout."""
        try:
            len_bytes = self.process.stdout.read(4)
            if not len_bytes:
                raise EOFError("MediaPipe worker closed stdout (likely crashed).")
            msg_len = struct.unpack('!I', len_bytes)[0]
            if msg_len == 0:
                raise EOFError("MediaPipe worker sent zero-length message.")
            msg_data = self.process.stdout.read(msg_len)
            if len(msg_data) < msg_len:
                raise EOFError("Failed to read full message from MediaPipe worker.")
            return msg_data
        except (IOError, struct.error, EOFError) as e:
            logger.error(f"Failed to read from MediaPipe worker (pipe broken): {e}")
            self._log_worker_error()
            raise e

    def get_alignment_and_pose(self, face_crop_bgr):
        """
        Sends a face crop to the MediaPipe worker.
        Returns: (aligned_face, yaw, pitch, roll) or (None, 0, 0, 0)
        """
        try:
            if face_crop_bgr is None or face_crop_bgr.size == 0:
                logger.warning("Skipping empty face crop.")
                return None, 0.0, 0.0, 0.0, 0.0
                
            data = pickle.dumps(face_crop_bgr)
            self._write_message(data)
            
            response_data = self._read_message()
            if response_data is None:
                return None, 0.0, 0.0, 0.0
                
            response = pickle.loads(response_data)
            
            if len(response) == 5 and isinstance(response[4], str):
                _aligned, _y, _p, _r, err_msg = response
                logger.warning(f"MediaPipe worker returned an error: {err_msg}")
                return None, 0.0, 0.0, 0.0, 0.0
                
            if len(response) == 5:
                aligned_face, yaw, pitch, roll, aspect_ratio = response
                return aligned_face, yaw, pitch, roll, aspect_ratio
            else: # Fallback for old 4-tuple
                aligned_face, yaw, pitch, roll = response
                return aligned_face, yaw, pitch, roll, 1.0 # Return "passing" ratio

        except Exception as e:
            logger.error(f"Error in MediaPipe communication: {e}", exc_info=True)
            raise e # Re-raise to stop the main loop

    def stop(self):
        if self.process:
            logger.info("Stopping MediaPipe worker...")
            try:
                if self.process.stdin: self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info("MediaPipe worker stopped.")
            except (IOError, subprocess.TimeoutExpired, BrokenPipeError, AttributeError):
                self.process.kill()
                logger.warning("MediaPipe worker force-killed.")