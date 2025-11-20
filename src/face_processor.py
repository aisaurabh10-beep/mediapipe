# src/face_processor.py
import os
import cv2
import numpy as np
import logging
import traceback

logger = logging.getLogger(__name__)

class FaceProcessor:
    """
    Face processing + embedding using InsightFace 'antelopev2' (preferred) with fallback to
    model_zoo variants and DeepFace. Keeps API compatible with your project:
      - calculate_blur_laplacian(image)
      - calculate_sobel_sharpness(image)
      - get_embedding(face_image_bgr) -> np.ndarray (L2-normalized) or None
      - apply_ema(current_embedding, prev_embedding, alpha) -> np.ndarray
      - _batch_embed (internal): returns list of embeddings for faces list
    """

    def __init__(self, config=None):
        self.config = config
        # Keep compatibility with build_reference_database which reads face_processor.arcface_model
        self.arcface_model = "ArcFace"

        self.backend = None         # "insightface_faceanalysis" | "insightface" | "deepface"
        self.model = None           # underlying model object
        self.device = "cpu"         # "gpu" or "cpu"

        # Try FaceAnalysis (stable API yielding embeddings)
        try:
            from insightface.app import FaceAnalysis
            logger.info("Attempting to load insightface FaceAnalysis('antelopev2') ...")
            fa = FaceAnalysis(name='antelopev2')
            try:
                fa.prepare(ctx_id=0, det_size=(320, 320))
                self.device = "gpu"
                logger.info("Loaded FaceAnalysis('antelopev2') on GPU (ctx_id=0).")
            except Exception:
                fa.prepare(ctx_id=-1, det_size=(320, 320))
                self.device = "cpu"
                logger.info("Loaded FaceAnalysis('antelopev2') on CPU (ctx_id=-1).")
            self.model = fa
            self.backend = "insightface_faceanalysis"
        except Exception as e_fa:
            logger.debug(f"FaceAnalysis load failed: {e_fa}. Falling back to model_zoo attempts.")

            # Fallback: try model_zoo.get_model variants (may return detector/retinaface or other objects)
            try:
                import insightface
                from insightface import model_zoo
                possible_names = ["antelopev2", "antelope", "antelope_v2", "antelope-v2"]
                loaded = False
                for name in possible_names:
                    try:
                        logger.info(f"Trying insightface.model_zoo.get_model('{name}') ...")
                        m = model_zoo.get_model(name)
                        try:
                            m.prepare(ctx_id=0)
                            self.device = "gpu"
                            logger.info(f"Loaded InsightFace model '{name}' on GPU (ctx_id=0).")
                        except Exception:
                            m.prepare(ctx_id=-1)
                            self.device = "cpu"
                            logger.info(f"Loaded InsightFace model '{name}' on CPU (ctx_id=-1).")
                        self.model = m
                        self.backend = "insightface"
                        loaded = True
                        break
                    except Exception as me:
                        logger.debug(f"model_zoo.get_model('{name}') failed: {me}")
                        continue

                if not loaded:
                    raise ImportError("insightface model_zoo returned no usable model.")
            except Exception as e:
                # Final fallback: DeepFace
                logger.warning(f"InsightFace load failed: {e}. Falling back to DeepFace.")
                try:
                    from deepface import DeepFace
                    # warmup DeepFace (non-fatal)
                    try:
                        DeepFace.represent(np.zeros((112,112,3), dtype=np.uint8),
                                           model_name="ArcFace",
                                           detector_backend='skip',
                                           enforce_detection=False)
                    except Exception:
                        pass
                    self.model = DeepFace
                    self.backend = "deepface"
                    self.device = "cpu"
                except Exception as e2:
                    logger.critical("No embedding backend available (insightface/DeepFace).")
                    raise RuntimeError("No face embedding backend available. Install insightface or deepface.") from e2

        logger.info(f"FaceProcessor initialized. backend={self.backend}, device={self.device}")

    # ----------------------
    # Image quality helpers
    # ----------------------
    def calculate_blur_laplacian(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_sobel_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        return np.mean(magnitude)

    # ----------------------
    # InsightFace / DeepFace embedding helpers
    # ----------------------
    def _insightface_faceanalysis_embed(self, face_bgr):
        """
        Preferred path: use FaceAnalysis.get(image_rgb) -> list of face dicts with 'embedding'.
        """
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            # FaceAnalysis.get accepts single image (returns list of detections)
            faces = self.model.get(face_rgb)
            if faces and isinstance(faces, (list, tuple)):
                info = faces[0]
                if isinstance(info, dict) and 'embedding' in info and info['embedding'] is not None:
                    emb = np.array(info['embedding'], dtype=np.float32)
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    return emb
            # try resized fallback
            face_resized = cv2.resize(face_rgb, (112,112), interpolation=cv2.INTER_LINEAR)
            faces = self.model.get(face_resized)
            if faces and isinstance(faces, (list, tuple)):
                info = faces[0]
                if isinstance(info, dict) and 'embedding' in info and info['embedding'] is not None:
                    emb = np.array(info['embedding'], dtype=np.float32)
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    return emb
            return None
        except Exception as e:
            logger.debug(f"FaceAnalysis.get() failed: {e}")
            return None

    def _insightface_modelzoo_embed_single(self, face_bgr):
        """
        Attempt multiple method names / call signatures for model_zoo objects.
        """
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            m = self.model
            # try common single-face functions
            for method_name in ("get_embedding", "get_feat", "get_feature", "get_representation", "get_feats", "get_feats_by_image"):
                if hasattr(m, method_name):
                    try:
                        out = getattr(m, method_name)(face_rgb)
                        if out is None:
                            continue
                        emb = np.array(out, dtype=np.float32)
                        emb = emb / (np.linalg.norm(emb) + 1e-8)
                        return emb
                    except Exception:
                        # try with resized input
                        try:
                            out = getattr(m, method_name)(cv2.resize(face_rgb, (112,112)))
                            if out is not None:
                                emb = np.array(out, dtype=np.float32)
                                emb = emb / (np.linalg.norm(emb) + 1e-8)
                                return emb
                        except Exception:
                            continue

            # try forward/infer/call
            for fn in ("infer", "forward", "__call__"):
                if hasattr(m, fn):
                    try:
                        out = getattr(m, fn)(face_rgb)
                        if out is None:
                            continue
                        emb = np.array(out, dtype=np.float32)
                        emb = emb / (np.linalg.norm(emb) + 1e-8)
                        return emb
                    except Exception:
                        continue

            return None
        except Exception as e:
            logger.debug(f"modelzoo embed single failed: {e}")
            return None

    def _deepface_embed_single(self, face_bgr):
        try:
            rep = self.model.represent(face_bgr, model_name="ArcFace", enforce_detection=False, detector_backend='skip')
            if not rep:
                return None
            emb = np.array(rep[0]['embedding'], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb
        except Exception as e:
            logger.warning(f"DeepFace embedding failed: {e}")
            return None

    def _insightface_batch_faceanalysis(self, faces_bgr_list):
        """
        Preferred batched path using FaceAnalysis.get on list of RGB images.
        Returns list of embeddings (or None) in same order.
        """
        try:
            rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces_bgr_list]
            faces_out = self.model.get(rgb_list)  # typically list of lists or list of dicts
            embs = []
            if isinstance(faces_out, (list, tuple)):
                for item in faces_out:
                    if not item:
                        embs.append(None)
                        continue
                    # item may be list of detections or a dict
                    first = item[0] if isinstance(item, (list, tuple)) else item
                    if isinstance(first, dict) and 'embedding' in first and first['embedding'] is not None:
                        a = np.array(first['embedding'], dtype=np.float32)
                        a = a / (np.linalg.norm(a) + 1e-8)
                        embs.append(a)
                    elif isinstance(first, (list, tuple, np.ndarray)):
                        a = np.array(first, dtype=np.float32)
                        a = a / (np.linalg.norm(a) + 1e-8)
                        embs.append(a)
                    else:
                        embs.append(None)
                return embs
        except Exception as e:
            logger.debug(f"FaceAnalysis batch get failed: {e}")
        # fallback to per-face single calls
        out = []
        for f in faces_bgr_list:
            out.append(self._insightface_embed_single(f))
        return out

    # generic batch wrapper
    def _batch_embed(self, faces_bgr_list):
        if not faces_bgr_list:
            return []
        if self.backend == "insightface_faceanalysis":
            return self._insightface_batch_faceanalysis(faces_bgr_list)
        elif self.backend == "insightface":
            # try batch methods if present, else iterate
            m = self.model
            for method_name in ("get_embeddings", "get_feats", "get_features", "get_representation_batch"):
                if hasattr(m, method_name):
                    try:
                        rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces_bgr_list]
                        out = getattr(m, method_name)(rgb_list)
                        embs = []
                        for a in out:
                            if a is None:
                                embs.append(None)
                            else:
                                a = np.array(a, dtype=np.float32)
                                a = a / (np.linalg.norm(a) + 1e-8)
                                embs.append(a)
                        return embs
                    except Exception:
                        continue
            # fallback iterate
            return [self._insightface_modelzoo_embed_single(f) for f in faces_bgr_list]
        else:
            # deepface fallback iteratively
            return [self._deepface_embed_single(f) for f in faces_bgr_list]

    # ----------------------
    # Public embedding API
    # ----------------------
    def get_embedding(self, face_image_bgr):
        """
        Compute L2-normalized embedding for single face (BGR uint8). Returns np.ndarray or None.
        """
        try:
            if self.backend == "insightface_faceanalysis":
                return self._insightface_faceanalysis_embed(face_image_bgr)
            elif self.backend == "insightface":
                return self._insightface_modelzoo_embed_single(face_image_bgr)
            else:
                return self._deepface_embed_single(face_image_bgr)
        except Exception as e:
            logger.warning(f"get_embedding error: {e}\n{traceback.format_exc()}")
            return None

    # ----------------------
    # EMA smoothing (same API)
    # ----------------------
    def apply_ema(self, current_embedding, prev_embedding, alpha):
        if current_embedding is None:
            return prev_embedding
        if prev_embedding is None:
            norm = np.linalg.norm(current_embedding) + 1e-8
            return current_embedding / norm
        smoothed = (alpha * current_embedding) + ((1.0 - alpha) * prev_embedding)
        norm = np.linalg.norm(smoothed) + 1e-8
        return smoothed / norm
