import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from typing import List
import pickle
import os
from shared_types import DetectionEvent, PipelineConfig


class FacePipeline:
    def __init__(self, config: PipelineConfig, enrolled_dir: str = "enrolled_faces"):
        self.config = config
        self.enrolled_dir = enrolled_dir
        os.makedirs(enrolled_dir, exist_ok=True)

        self.face_app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection", "recognition"]
        )
        self.face_app.prepare(ctx_id=-1, det_size=(320, 320))

        self.enrolled_embeddings: List[np.ndarray] = []
        self._load_enrolled_faces()
        print(f"[FacePipeline] Loaded {len(self.enrolled_embeddings)} enrolled face(s)")

    def _load_enrolled_faces(self):
        self.enrolled_embeddings = []
        for f in os.listdir(self.enrolled_dir):
            if f.endswith(".pkl"):
                with open(os.path.join(self.enrolled_dir, f), "rb") as fp:
                    self.enrolled_embeddings.append(pickle.load(fp))

    def enroll_from_embeddings(self, embeddings: List[np.ndarray], name: str = "owner"):
        """Average a list of embeddings collected from webcam and save."""
        if not embeddings:
            raise ValueError("No embeddings to enroll")
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        save_path = os.path.join(self.enrolled_dir, f"{name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(mean_emb, f)
        self._load_enrolled_faces()
        print(f"[FacePipeline] Enrolled '{name}' from {len(embeddings)} frames")

    def enroll_face(self, image_paths: List[str], name: str = "owner"):
        embeddings = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            faces = self.face_app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings.append(face.embedding)
        if not embeddings:
            raise ValueError("No faces found in any image")
        self.enroll_from_embeddings(embeddings, name)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _is_whitelisted(self, embedding: np.ndarray) -> bool:
        for enrolled in self.enrolled_embeddings:
            if self._cosine_similarity(embedding, enrolled) >= self.config.face_similarity_threshold:
                return True
        return False

    def detect_faces(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """
        Single method — runs InsightFace detection + recognition every call.
        """
        faces = self.face_app.get(frame)
        events = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            is_whitelisted = self._is_whitelisted(face.embedding) if self.enrolled_embeddings else False
            events.append(DetectionEvent(
                type="face",
                bbox=(x1, y1, x2, y2),
                confidence=float(face.det_score),
                frame_id=frame_id,
                blur=not is_whitelisted
            ))
        return events