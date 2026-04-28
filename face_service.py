import numpy as np
import cv2
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FotoKash-FaceAI")


class FaceService:
    def __init__(self):
        logger.info("Chargement du modele InsightFace...")
        from insightface.app import FaceAnalysis
        
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Supprimer le cache buffalo_l s'il existe
        import shutil
        old_model = os.path.join(model_dir, "models", "buffalo_l")
        if os.path.exists(old_model):
            shutil.rmtree(old_model)
            logger.info("Cache buffalo_l supprime")
        
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("Modele charge avec succes !")

    def detect_faces(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Impossible de decoder l'image.")
        
        faces = self.app.get(img)
        results = []
        h, w = img.shape[:2]
        
        for face in faces:
            bbox = face.bbox.tolist()
            results.append({
                "bbox": {
                    "x": max(0, bbox[0] / w),
                    "y": max(0, bbox[1] / h),
                    "w": min(1, (bbox[2] - bbox[0]) / w),
                    "h": min(1, (bbox[3] - bbox[1]) / h),
                },
                "embedding": face.embedding.tolist(),
                "confidence": round(float(face.det_score), 4),
            })
        
        logger.info(f"{len(results)} visage(s) detecte(s)")
        return results

    def extract_single_embedding(self, image_bytes):
        faces = self.detect_faces(image_bytes)
        if not faces:
            return None
        best_face = max(faces, key=lambda f: f["confidence"])
        if best_face["confidence"] < 0.5:
            return None
        return best_face["embedding"]

    def compare_embeddings(self, embedding1, embedding2):
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(max(0, similarity))

    def cluster_faces(self, embeddings, threshold=0.5):
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        from sklearn.cluster import DBSCAN
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / norms
        clustering = DBSCAN(eps=threshold, min_samples=2, metric="cosine").fit(normalized)
        return clustering.labels_.tolist()