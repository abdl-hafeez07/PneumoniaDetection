from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from PIL import Image
import numpy as np
import os


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
# LOAD MODEL
# ============================================================

print("=" * 60)
print("CHEST X-RAY PNEUMONIA DETECTION")
print("=" * 60)

print("Model path:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

print("Loading EfficientNetB0 model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# Warm up model to pre-compile graph during boot, preventing request timeouts
try:
    print("Warming up model graph...")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy_input = preprocess_input(dummy_input)
    model.predict(dummy_input, verbose=0)
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
        # Open image using PIL
        # ----------------------------------------------------

        print("Opening image...")

        if hasattr(file.stream, "seek"):
            file.stream.seek(0)

        image = Image.open(file.stream)


        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        image = image.convert("RGB")


        print(
            "Original image size:",
            image.size
        )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            IMAGE_SIZE
        )


        print(
            "Resized image:",
            image.size
        )


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        )


        print(
            "Array shape before batch:",
            image_array.shape
        )


        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        print(
            "Array shape after batch:",
            image_array.shape
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # SAME EfficientNetB0 preprocessing used
        # during training
        # ----------------------------------------------------

        image_array = preprocess_input(
            image_array
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        print("Running model prediction...")

        prediction = model.predict(
            image_array,
            verbose=0
        )


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