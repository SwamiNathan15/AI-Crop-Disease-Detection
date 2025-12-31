from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
model = tf.keras.models.load_model("tomato_disease_model.h5")

# Class names (same order as training)
class_names = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato_healthy'
]

@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")  # Match the form input name
        if file and file.filename:
            # Save file
            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Load and preprocess image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]
            confidence = int(round(float(np.max(preds) * 100)))
            
            # Convert file path to URL for template
            image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_url
    )

if __name__ == "__main__":
    app.run(debug=True)
