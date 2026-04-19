# FotoKash Face AI — Micro-service de reconnaissance faciale

Service Python qui détecte les visages et extrait des empreintes faciales 512D
grâce au modèle ArcFace (InsightFace). Communique avec le back-end Node.js
via API REST.

## Installation sur Windows

### 1. Installer Python 3.10+

Téléchargez Python depuis : https://www.python.org/downloads/

**IMPORTANT** : Cochez la case "Add Python to PATH" pendant l'installation !

Vérifiez dans PowerShell :
```
python --version
pip --version
```

### 2. Installer les dépendances

```
cd fotokash-face-service
pip install -r requirements.txt
```

Note : l'installation de insightface et onnxruntime peut prendre quelques
minutes. Si vous avez une erreur avec Visual C++, installez les
"Microsoft C++ Build Tools" depuis :
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 3. Télécharger le modèle ArcFace (~300 Mo, une seule fois)

```
python download_model.py
```

### 4. Lancer le service

```
python app.py
```

Le service démarre sur http://localhost:5000

Vérifiez : http://localhost:5000/health

## Routes API

| Méthode | Route              | Description                           |
|---------|--------------------|---------------------------------------|
| GET     | /health            | Statut du service                     |
| POST    | /detect-faces      | Détecter visages dans une photo       |
| POST    | /extract-embedding | Extraire empreinte d'un selfie        |
| POST    | /compare           | Comparer deux empreintes              |
| POST    | /cluster           | Regrouper visages similaires          |
| POST    | /batch-detect      | Détection en masse (jusqu'à 50 imgs)  |

## Exemple d'utilisation

### Détecter les visages dans une photo
```
curl -X POST http://localhost:5000/detect-faces \
  -F "image=@photo.jpg"
```

### Extraire l'empreinte d'un selfie
```
curl -X POST http://localhost:5000/extract-embedding \
  -F "image=@selfie.jpg"
```

## Fonctionnement avec le back-end Node.js

Le back-end Node.js appelle ce service automatiquement :

1. **Upload photo** → Node.js envoie l'image à `/detect-faces`
   → reçoit les embeddings → les stocke dans PostgreSQL (pgvector)

2. **Selfie client** → Node.js envoie le selfie à `/extract-embedding`
   → reçoit le vecteur → fait une recherche vectorielle dans pgvector
   → retourne les photos correspondantes

Le fichier `.env` du back-end doit contenir :
```
FACE_AI_SERVICE_URL=http://localhost:5000
```
