import base64
from io import BytesIO

from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify
from models.K_Nearest_Neighbors.KNN_Init import KNN_init

model = KNN_init()
app = Flask(__name__)


def preprocess_image(image_data):
    # 1. Remove base64 header and decode
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("L")

    # 2. Resize to MNIST format
    image = image.resize((28, 28))

    # 3. Convert to numpy
    img_array = np.array(image)

    # 4. MATCHING OPENML FORMAT:
    # A) Do NOT divide by 255.0.
    # fetch_openml('mnist_784') returns integers/floats from 0-255.

    # B) Flatten the array.
    # KNN doesn't see a "grid" (28,28,1), it sees a list of 784 features.
    img_array = img_array.reshape(1, 784)

    return img_array


@app.route("/")
def home():
    return render_template("/index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    image_data = data["image"]
    model_name = data["model"]

    img = preprocess_image(image_data)

    prediction = model.predict(img)
    # digit = int(np.argmax(prediction))
    digit = prediction[0]
    # confidence = float(np.max(prediction))
    confidence = 0.90

    return jsonify({
        "digit": int(digit),
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)