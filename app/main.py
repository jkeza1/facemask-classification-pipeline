import os
import uuid
from flask import Flask, request, jsonify, render_template
from .model import load_model_from_file, predict_image, retrain_model  # Adjust import as needed

app = Flask(__name__)

# Load your model once at startup
model = load_model_from_file()

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template("index.html", prediction="No image file provided")

        img_file = request.files['image']
        if img_file.filename == '':
            return render_template("index.html", prediction="No selected file")

        # Save uploaded file with a unique name
        filename = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(img_path)

        prediction = predict_image(model, img_path)

        # Optionally remove file after prediction
        # os.remove(img_path)

    return render_template("index.html", prediction=prediction)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img_file.save(img_path)

    prediction = predict_image(model, img_path)

    # Optionally remove file after prediction
    # os.remove(img_path)

    return jsonify({'prediction': prediction})


@app.route('/retrain', methods=['POST'])
def retrain():
    new_data_path = request.form.get('data_path')

    if not new_data_path or not os.path.exists(new_data_path):
        return render_template("index.html", prediction="Invalid or missing data path for retraining")

    retrain_model(new_data_path)
    return render_template("index.html", prediction="Model retrained successfully!")


if __name__ == '__main__':
    app.run(debug=True)
