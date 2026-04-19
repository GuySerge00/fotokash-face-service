from flask import Flask, request, jsonify
from flask_cors import CORS
from face_service import FaceService
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FotoKash-API")

app = Flask(__name__)
CORS(app)

# Charger le modèle au démarrage (une seule fois)
logger.info("Initialisation du service facial...")
face_ai = FaceService()
logger.info("Service facial prêt !")

# Taille max des uploads : 25 Mo
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ===== ROUTES =====


@app.route("/health", methods=["GET"])
def health():
    """Vérifier que le service est opérationnel."""
    return jsonify(
        {
            "status": "ok",
            "service": "FotoKash Face AI",
            "model": "buffalo_l (ArcFace)",
            "embedding_size": 512,
        }
    )


@app.route("/detect-faces", methods=["POST"])
def detect_faces():
    """
    Détecter tous les visages dans une image.
    Utilisé lors de l'upload par le photographe.

    Input: image file (multipart/form-data)
    Output: liste de visages avec bbox, embedding 512D, confidence
    """
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    if len(image_bytes) == 0:
        return jsonify({"error": "Image vide."}), 400

    try:
        faces = face_ai.detect_faces(image_bytes)

        elapsed = round(time.time() - start, 3)
        logger.info(
            f"Détection: {len(faces)} visage(s) en {elapsed}s"
        )

        return jsonify(
            {
                "faces": faces,
                "count": len(faces),
                "processing_time_ms": int(elapsed * 1000),
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur détection: {e}")
        return jsonify({"error": "Erreur de traitement."}), 500


@app.route("/extract-embedding", methods=["POST"])
def extract_embedding():
    """
    Extraire l'empreinte faciale d'un selfie.
    Utilisé côté client pour la recherche par selfie.

    Input: image file (selfie, multipart/form-data)
    Output: vecteur 512D du visage principal
    """
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    if len(image_bytes) == 0:
        return jsonify({"error": "Image vide."}), 400

    try:
        embedding = face_ai.extract_single_embedding(image_bytes)

        if embedding is None:
            return jsonify(
                {
                    "error": "Aucun visage détecté. "
                    "Assurez-vous d'être bien éclairé "
                    "et face à la caméra.",
                    "embedding": None,
                }
            ), 400

        elapsed = round(time.time() - start, 3)
        logger.info(f"Extraction selfie en {elapsed}s")

        return jsonify(
            {
                "embedding": embedding,
                "dimensions": len(embedding),
                "processing_time_ms": int(elapsed * 1000),
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur extraction: {e}")
        return jsonify({"error": "Erreur de traitement."}), 500


@app.route("/compare", methods=["POST"])
def compare_faces():
    """
    Comparer deux empreintes faciales.
    Utile pour le debug et les tests.

    Input JSON: { "embedding1": [...], "embedding2": [...] }
    Output: score de similarité entre 0 et 1
    """
    data = request.get_json()

    if not data or "embedding1" not in data or "embedding2" not in data:
        return jsonify(
            {"error": "Deux embeddings requis (embedding1, embedding2)."}
        ), 400

    try:
        similarity = face_ai.compare_embeddings(
            data["embedding1"], data["embedding2"]
        )

        # Seuil de 0.85 pour considérer que c'est la même personne
        is_same_person = similarity >= 0.85

        return jsonify(
            {
                "similarity": round(similarity, 4),
                "is_same_person": is_same_person,
                "threshold": 0.85,
            }
        )

    except Exception as e:
        logger.error(f"Erreur comparaison: {e}")
        return jsonify({"error": "Erreur de comparaison."}), 500


@app.route("/cluster", methods=["POST"])
def cluster_faces():
    """
    Regrouper des visages en clusters (une personne = un cluster).
    Utilisé après l'upload en masse pour organiser les photos.

    Input JSON: { "embeddings": [[...], [...], ...], "threshold": 0.5 }
    Output: liste d'IDs de clusters
    """
    data = request.get_json()

    if not data or "embeddings" not in data:
        return jsonify({"error": "Liste d'embeddings requise."}), 400

    embeddings = data["embeddings"]
    threshold = data.get("threshold", 0.5)

    if len(embeddings) == 0:
        return jsonify({"error": "Liste d'embeddings vide."}), 400

    try:
        labels = face_ai.cluster_faces(embeddings, threshold)
        n_clusters = len(set(labels) - {-1})

        return jsonify(
            {
                "labels": labels,
                "n_clusters": n_clusters,
                "n_faces": len(embeddings),
            }
        )

    except Exception as e:
        logger.error(f"Erreur clustering: {e}")
        return jsonify({"error": "Erreur de clustering."}), 500


@app.route("/batch-detect", methods=["POST"])
def batch_detect():
    """
    Détecter les visages dans plusieurs images en une requête.
    Utilisé lors de l'upload en masse par le photographe.

    Input: plusieurs fichiers image (multipart/form-data, champ "images")
    Output: résultats par image
    """
    start = time.time()

    if "images" not in request.files:
        return jsonify({"error": "Aucune image envoyée."}), 400

    files = request.files.getlist("images")

    if len(files) == 0:
        return jsonify({"error": "Aucun fichier reçu."}), 400

    if len(files) > 50:
        return jsonify({"error": "Maximum 50 images par requête."}), 400

    results = []
    total_faces = 0

    for i, file in enumerate(files):
        try:
            image_bytes = file.read()
            faces = face_ai.detect_faces(image_bytes)
            total_faces += len(faces)

            results.append(
                {
                    "filename": file.filename or f"image_{i}",
                    "faces": faces,
                    "count": len(faces),
                    "status": "ok",
                }
            )
        except Exception as e:
            results.append(
                {
                    "filename": file.filename or f"image_{i}",
                    "faces": [],
                    "count": 0,
                    "status": "error",
                    "error": str(e),
                }
            )

    elapsed = round(time.time() - start, 3)
    logger.info(
        f"Batch: {total_faces} visage(s) dans "
        f"{len(files)} image(s) en {elapsed}s"
    )

    return jsonify(
        {
            "results": results,
            "total_images": len(files),
            "total_faces": total_faces,
            "processing_time_ms": int(elapsed * 1000),
        }
    )


# ===== GESTION DES ERREURS =====


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Image trop volumineuse. Maximum 25 Mo."}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Erreur serveur interne."}), 500


# ===== DÉMARRAGE =====

if __name__ == "__main__":
    print(
        """
    ╔═══════════════════════════════════════════╗
    ║       FotoKash Face AI Service            ║
    ║       Port: 5000                          ║
    ║       Modèle: ArcFace (buffalo_l)         ║
    ║       Embedding: 512 dimensions           ║
    ╚═══════════════════════════════════════════╝

    Routes disponibles :
    • GET  /health            → Statut du service
    • POST /detect-faces      → Détecter visages (upload photo)
    • POST /extract-embedding → Extraire empreinte (selfie client)
    • POST /compare           → Comparer deux empreintes
    • POST /cluster           → Regrouper visages similaires
    • POST /batch-detect      → Détection en masse (multi-images)
    """
    )

    app.run(host="0.0.0.0", port=5000, debug=False)
