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
    retrain_message = request.args.get('retrain_message')
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

    return render_template("index.html", prediction=prediction, retrain_message=retrain_message)


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
from flask import redirect, url_for

@app.route('/retrain', methods=['POST'])
def retrain():
    if 'training_data' not in request.files:
        return redirect(url_for('home'))

    files = request.files.getlist('training_data')
    if not files or files[0].filename == '':
        return redirect(url_for('home'))

    retrain_folder = os.path.join(UPLOAD_FOLDER, f"retrain_{uuid.uuid4().hex}")
    os.makedirs(retrain_folder, exist_ok=True)

    total_files = 0
    for file in files:
        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = f"{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(retrain_folder, filename)
            file.save(file_path)
            total_files += 1

    if total_files == 0:
        return redirect(url_for('home'))

    retrain_model(retrain_folder)

    # No message, just redirect
    return redirect(url_for('home'))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
