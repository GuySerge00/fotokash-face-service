import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FotoKash-FaceAI")


class FaceService:
    """
    Service de reconnaissance faciale FotoKash.
    Utilise InsightFace (modèle ArcFace) pour :
    - Détecter les visages dans une image
    - Extraire un vecteur d'empreinte 512D par visage
    - Regrouper les visages similaires (clustering)
    """

    def __init__(self):
        logger.info("Chargement du modèle InsightFace (ArcFace)...")

        # Initialiser InsightFace avec le modèle buffalo_l (inclut ArcFace)
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.join(os.path.dirname(__file__), "models"),
            providers=["CPUExecutionProvider"],  # GPU: ["CUDAExecutionProvider"]
        )

        # Préparer le modèle — taille d'entrée 640x640
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info("Modèle chargé avec succès !")

    def detect_faces(self, image_bytes):
        """
        Détecte tous les visages dans une image.

        Args:
            image_bytes: Image en bytes (JPEG, PNG)

        Returns:
            Liste de dictionnaires avec bbox, embedding, confidence
        """
        # Convertir les bytes en image OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Impossible de décoder l'image.")

        # Détecter les visages
        faces = self.app.get(img)

        results = []
        for face in faces:
            bbox = face.bbox.tolist()  # [x1, y1, x2, y2]
            embedding = face.embedding.tolist()  # Vecteur 512D
            confidence = float(face.det_score)

            # Normaliser les coordonnées de la bounding box
            h, w = img.shape[:2]
            normalized_bbox = {
                "x": max(0, bbox[0] / w),
                "y": max(0, bbox[1] / h),
                "w": min(1, (bbox[2] - bbox[0]) / w),
                "h": min(1, (bbox[3] - bbox[1]) / h),
            }

            results.append(
                {
                    "bbox": normalized_bbox,
                    "embedding": embedding,
                    "confidence": round(confidence, 4),
                }
            )

        logger.info(f"{len(results)} visage(s) détecté(s)")
        return results

    def extract_single_embedding(self, image_bytes):
        """
        Extrait l'empreinte d'un seul visage (pour le selfie client).
        Retourne le visage avec le plus haut score de confiance.

        Args:
            image_bytes: Image selfie en bytes

        Returns:
            Vecteur 512D ou None si aucun visage détecté
        """
        faces = self.detect_faces(image_bytes)

        if not faces:
            return None

        # Prendre le visage avec la meilleure confiance
        best_face = max(faces, key=lambda f: f["confidence"])

        if best_face["confidence"] < 0.5:
            logger.warning(
                f"Confiance trop basse: {best_face['confidence']}"
            )
            return None

        return best_face["embedding"]

    def compare_embeddings(self, embedding1, embedding2):
        """
        Calcule la similarité cosinus entre deux empreintes faciales.

        Args:
            embedding1: Vecteur 512D (liste)
            embedding2: Vecteur 512D (liste)

        Returns:
            Score de similarité entre 0 et 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Similarité cosinus
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

        return float(max(0, similarity))

    def cluster_faces(self, embeddings, threshold=0.5):
        """
        Regroupe les visages similaires en clusters.
        Chaque cluster = une personne unique.

        Args:
            embeddings: Liste de vecteurs 512D
            threshold: Distance max pour considérer deux visages identiques

        Returns:
            Liste d'IDs de clusters (-1 = bruit/non classé)
        """
        if len(embeddings) < 2:
            return [0] * len(embeddings)

        embeddings_array = np.array(embeddings)

        # Normaliser les embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized = embeddings_array / norms

        # DBSCAN avec distance cosinus
        clustering = DBSCAN(
            eps=threshold,
            min_samples=2,
            metric="cosine",
        ).fit(normalized)

        labels = clustering.labels_.tolist()
        n_clusters = len(set(labels) - {-1})
        logger.info(
            f"Clustering: {n_clusters} personne(s) identifiée(s) "
            f"parmi {len(embeddings)} visages"
        )

        return labels
