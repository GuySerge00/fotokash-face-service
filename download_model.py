"""
Script pour télécharger le modèle InsightFace buffalo_l.
À exécuter une seule fois avant de lancer le service.
"""

import os
import insightface
from insightface.app import FaceAnalysis

def download_model():
    print("=" * 50)
    print("FotoKash — Téléchargement du modèle ArcFace")
    print("=" * 50)
    print()
    print("Ce téléchargement ne se fait qu'une seule fois.")
    print("Taille approximative : ~300 Mo")
    print()

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Téléchargement en cours...")
    app = FaceAnalysis(
        name="buffalo_l",
        root=models_dir,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    print()
    print("Modèle téléchargé avec succès !")
    print(f"Emplacement : {models_dir}")
    print()

    # Test rapide
    print("Test rapide du modèle...")
    import numpy as np
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = app.get(test_img)
    print(f"Test OK — {len(faces)} visage(s) détecté(s) (normal: 0 sur image vide)")
    print()
    print("Le service est prêt à être lancé avec : python app.py")


if __name__ == "__main__":
    download_model()
