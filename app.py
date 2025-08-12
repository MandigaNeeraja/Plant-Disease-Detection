from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load trained model
model = load_model("plant_disease_model.h5")

# Class labels (same order as used in training)
class_labels = ['Black Rot','ESCA','Healthy','Leaf Blight']

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(image_path)

        # Load & preprocess the image
        image = load_img(image_path, target_size=(224,224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Convert to batch
        image = image / 255.0  # Normalize

        # Make prediction
        prediction = model.predict(image)
        predicted_class = class_labels[np.argmax(prediction)]

        return render_template('index.html', prediction=predicted_class, image_path=image_path)

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
