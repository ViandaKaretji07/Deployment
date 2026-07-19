"""
Deteksi Bahasa Isyarat - Backend Flask untuk Render
=====================================================
Endpoint:
  GET  /                -> halaman utama (webcam UI)
  POST /predict          -> menerima 1 frame (base64 JPEG/PNG), mengembalikan
                             prediksi huruf/angka + confidence
  POST /speak             -> menerima teks kalimat, mengembalikan audio MP3 (base64)

Model & artefak dibaca dari folder model/ (hasil ekstrak model_artifacts_isl.zip
yang dihasilkan notebook Colab):
  model/model.joblib atau model/model.keras
  model/scaler.joblib
  model/label_encoder.joblib
  model/model_info.json
"""

import os
import io
import json
import base64
import urllib.request

import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_file

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from gtts import gTTS

# ---------------------------------------------------------------------------
# Konfigurasi & load artefak
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LANDMARKER_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_landmarker_model():
    """Unduh hand_landmarker.task jika belum ada di server. Ini jaring pengaman
    kalau langkah wget di Build Command tidak berjalan (mis. Build Command di
    Render dashboard belum diedit sesuai README)."""
    if os.path.exists(LANDMARKER_TASK_PATH) and os.path.getsize(LANDMARKER_TASK_PATH) > 0:
        return
    print("hand_landmarker.task tidak ditemukan, mengunduh dari Google...")
    urllib.request.urlretrieve(LANDMARKER_TASK_URL, LANDMARKER_TASK_PATH)
    print(f"Berhasil diunduh ke {LANDMARKER_TASK_PATH}")


ensure_hand_landmarker_model()

with open(os.path.join(MODEL_DIR, "model_info.json"), "r") as f:
    MODEL_INFO = json.load(f)

MODEL_TYPE = MODEL_INFO["model_type"]  # "sklearn" atau "keras"
CLASSES = MODEL_INFO["classes"]
N_FEATURES = MODEL_INFO["n_features"]

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

if MODEL_TYPE == "keras":
    # Import tensorflow hanya jika benar-benar dibutuhkan (hemat memori jika model sklearn)
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.keras"))
else:
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))

# MediaPipe HandLandmarker (Tasks API)
base_options = BaseOptions(model_asset_path=LANDMARKER_TASK_PATH)
landmarker_options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=RunningMode.IMAGE,
)
hand_landmarker = HandLandmarker.create_from_options(landmarker_options)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
def extract_landmarks(image_bgr):
    """Sama persis dengan preprocessing di notebook: 63 fitur (21 landmark x xyz),
    ternormalisasi translasi (relatif wrist) dan skala (dibagi jarak maksimum)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = hand_landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand_landmarks = result.hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    wrist = coords[0]
    coords = coords - wrist

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist

    return coords.flatten()


def decode_base64_image(data_url):
    """Mengubah string base64 (data URL dari <canvas>.toDataURL()) menjadi citra BGR (OpenCV)."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_bgr


def predict_from_features(feat):
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    if MODEL_TYPE == "keras":
        proba = model.predict(feat_scaled, verbose=0)[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
    else:
        idx = int(model.predict(feat_scaled)[0])
        if hasattr(model, "predict_proba"):
            conf = float(np.max(model.predict_proba(feat_scaled)))
        else:
            conf = 1.0
    label = label_encoder.inverse_transform([idx])[0]
    return label, conf


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES, model_name=MODEL_INFO.get("model_name", ""))


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Field 'image' (base64) wajib diisi"}), 400

    try:
        image_bgr = decode_base64_image(payload["image"])
        if image_bgr is None:
            return jsonify({"error": "Gagal decode citra"}), 400

        feat = extract_landmarks(image_bgr)
        if feat is None:
            return jsonify({"detected": False, "label": None, "confidence": 0.0})

        label, conf = predict_from_features(feat)
        return jsonify({"detected": True, "label": str(label), "confidence": conf})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    payload = request.get_json(silent=True)
    text = (payload or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' wajib diisi"}), 400

    try:
        tts = gTTS(text=text, lang="id")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return jsonify({"audio_base64": audio_b64, "mime": "audio/mpeg"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_INFO.get("model_name"), "classes": len(CLASSES)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
