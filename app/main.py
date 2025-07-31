import os
import uuid
from flask import Flask, request, jsonify, render_template
from model import load_model_from_file, predict_image, retrain_model

app = Flask(__name__)

# Load your model once at startup
model = load_model_from_file()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
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
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        img_file = request.files['image']
        if img_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)

        print(f"[DEBUG] Saving uploaded image to {img_path}")
        img_file.save(img_path)

        print(f"[DEBUG] Saved uploaded image to {img_path}")

        prediction = predict_image(model, img_path)

        # Optionally remove file after prediction
        # os.remove(img_path)

        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    new_data_path = request.form.get('data_path')

    if not new_data_path or not os.path.exists(new_data_path):
        return render_template("index.html", prediction="Invalid or missing data path for retraining")

    retrain_model(new_data_path)
    return render_template("index.html", prediction="Model retrained successfully!")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
