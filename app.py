import os

# Optimize TensorFlow for cloud containers (prevents CPU thread thrashing and memory bloat)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf

# Disable XLA compilation (XLA compilation takes 90s+ on shared cloud CPUs, causing worker timeouts)
tf.config.optimizer.set_jit(False)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import gc
from PIL import Image
import numpy as np


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFLITE_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model.tflite"
)

if not os.path.exists(TFLITE_PATH):
    TFLITE_PATH = os.path.join(
        BASE_DIR,
        "model.tflite"
    )

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "EfficientNetB0_Pneumonia_Final.keras"
)

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(
        BASE_DIR,
        "EfficientNetB0_Pneumonia_Final.keras"
    )

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL (TFLite for lightweight RAM or Keras fallback)
# ============================================================

print("=" * 60)
print("CHEST X-RAY PNEUMONIA DETECTION")
print("=" * 60)

tflite_interpreter = None
tflite_input_idx = None
tflite_output_idx = None
model = None

if os.path.exists(TFLITE_PATH):
    print("Loading optimized TFLite model:")
    print(TFLITE_PATH)
    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    tflite_interpreter.allocate_tensors()
    tflite_input_idx = tflite_interpreter.get_input_details()[0]["index"]
    tflite_output_idx = tflite_interpreter.get_output_details()[0]["index"]
    print("TFLite model loaded successfully! (RAM: ~50MB)")
elif os.path.exists(MODEL_PATH):
    print("Loading EfficientNetB0 Keras model:")
    print(MODEL_PATH)
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(
        f"Neither TFLite ({TFLITE_PATH}) nor Keras ({MODEL_PATH}) model was found!"
    )

# Warm up model during boot to prevent first-request latency
try:
    print("Warming up model...")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy_input = preprocess_input(dummy_input)
    if tflite_interpreter is not None:
        tflite_interpreter.set_tensor(tflite_input_idx, dummy_input)
        tflite_interpreter.invoke()
    else:
        model(dummy_input, training=False)
    print("Model warmup complete!")
except Exception as e:
    print("Warmup notice:", e)

print("=" * 60)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model": "EfficientNetB0",
        "model_loaded": True
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    print("\n" + "=" * 60)
    print("NEW PREDICTION REQUEST")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "file" not in request.files:

            print("ERROR: No file in request")

            return jsonify({
                "error": "No image file was uploaded."
            }), 400


        file = request.files["file"]


        if file.filename == "":

            print("ERROR: Empty filename")

            return jsonify({
                "error": "Please select an image."
            }), 400


        print("Filename:", file.filename)


        # ----------------------------------------------------
        # Check extension
        # ----------------------------------------------------

        allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png"
        }

        extension = os.path.splitext(
            file.filename
        )[1].lower()


        if extension not in allowed_extensions:

            print("ERROR: Invalid extension")

            return jsonify({
                "error": "Only JPG, JPEG and PNG images are supported."
            }), 400


        # ----------------------------------------------------
        # Open image using PIL with memory-safe drafting
        # ----------------------------------------------------

        print("Opening image...")

        if hasattr(file.stream, "seek"):
            file.stream.seek(0)

        image = Image.open(file.stream)

        # Optimize memory during JPEG decode of high-resolution images
        if hasattr(image, "draft"):
            try:
                image.draft("RGB", IMAGE_SIZE)
            except Exception:
                pass

        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # EfficientNetB0 Preprocessing
        # ----------------------------------------------------

        image_array = preprocess_input(
            image_array
        )

        # ----------------------------------------------------
        # Prediction (Fast & memory-efficient)
        # ----------------------------------------------------

        print("Running model prediction...")

        if tflite_interpreter is not None:
            tflite_interpreter.set_tensor(tflite_input_idx, image_array)
            tflite_interpreter.invoke()
            prediction = tflite_interpreter.get_tensor(tflite_output_idx)
        else:
            prediction = model(
                image_array,
                training=False
            ).numpy()


        print(
            "Raw model output:",
            prediction
        )


        # ----------------------------------------------------
        # Extract pneumonia probability
        # ----------------------------------------------------

        pneumonia_probability = float(
            prediction[0][0]
        )


        # Keep probability valid
        pneumonia_probability = max(
            0.0,
            min(
                1.0,
                pneumonia_probability
            )
        )


        normal_probability = (
            1.0 - pneumonia_probability
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if pneumonia_probability >= 0.5:

            predicted_class = "PNEUMONIA"
            confidence = pneumonia_probability

        else:

            predicted_class = "NORMAL"
            confidence = normal_probability


        # ----------------------------------------------------
        # Percentages
        # ----------------------------------------------------

        normal_percentage = (
            normal_probability * 100
        )

        pneumonia_percentage = (
            pneumonia_probability * 100
        )

        confidence_percentage = (
            confidence * 100
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

            "predicted_class": predicted_class,

            "confidence": round(
                confidence_percentage,
                2
            ),

            "normal_probability": round(
                normal_percentage,
                2
            ),

            "pneumonia_probability": round(
                pneumonia_percentage,
                2
            )
        }


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("-" * 60)
        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            f"{confidence_percentage:.2f}%"
        )

        print(
            "Normal:",
            f"{normal_percentage:.2f}%"
        )

        print(
            "Pneumonia:",
            f"{pneumonia_percentage:.2f}%"
        )

        print("-" * 60)
        print("Prediction successful!")
        print("=" * 60)


        return jsonify(result)


    except Exception as e:

        # ----------------------------------------------------
        # IMPORTANT:
        # Print the REAL error in terminal
        # ----------------------------------------------------

        print("\n!!! PREDICTION ERROR !!!")
        print(type(e).__name__)
        print(str(e))
        print("=" * 60)


        return jsonify({
            "error": str(e)
        }), 500

    finally:
        gc.collect()


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image is too large. Maximum allowed size is 10 MB."
    }), 413


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("Server starting...")
    print(f"http://localhost:{port}")
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )