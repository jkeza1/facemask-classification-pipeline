import os
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from model import load_model_from_file, predict_image, retrain_model

app = Flask(__name__)

# Load model once at startup
model = load_model_from_file()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

details_map = {
    "with_mask": "✅ The person is wearing a mask properly, covering both nose and mouth.",
    "without_mask": "⚠️ No mask detected. Please wear a mask to protect yourself and others.",
    "mask_weared_incorrect": "⚠️ The mask is worn incorrectly. Please cover both nose and mouth properly."
}

def verify_label_mapping():
    """Ensure model outputs match our details_map keys"""
    test_cases = [
        ("with_mask", True),
        ("WITH_MASK", False),  # Should fail on case
        ("without_mask", True),
        ("mask_weared_incorrect", True),
        ("mask_worn_incorrect", False)  # Should fail
    ]
    
    for label, should_pass in test_cases:
        normalized = label.strip()  # No lower() - case sensitive!
        result = normalized in details_map
        print(f"Test '{label}': {'✅' if result == should_pass else '❌'} "
              f"(Expected {'match' if should_pass else 'no match'})")

verify_label_mapping()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_path = None
    details = None
    retrain_message = request.args.get("retrain_message", None)
    retrained = request.args.get('retrained', '0')

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', retrain_message=None, retrained=None)

        img_file = request.files['image']
        if img_file.filename == '':
            return render_template("index.html", prediction="No selected file", retrained=retrained)

        filename = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img_file.save(img_path)

        prediction = predict_image(model, img_path)

        # Normalize prediction label for details map lookup
        if isinstance(prediction, dict):
            label = prediction.get('label', '').strip().lower()
        elif isinstance(prediction, str):
            label = prediction.strip().lower()
        else:
            label = str(prediction).strip().lower()

        details = details_map.get(label, "❓ No additional details available.")
        image_path = f"/uploads/{filename}"

    return render_template("index.html",
                           prediction=prediction,
                           image_path=image_path,
                           details=details,
                           retrain_message=retrain_message,
                           retrained=retrained)


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
        img_file.save(img_path)

        prediction = predict_image(model, img_path)

        # Normalize prediction label for details map lookup
        if isinstance(prediction, dict):
            label = prediction.get('label', '').strip().lower()
        elif isinstance(prediction, str):
            label = prediction.strip().lower()
        else:
            label = str(prediction).strip().lower()

        details = details_map.get(label, "❓ No additional details available.")

        return render_template("index.html",
                               prediction=label,
                               details=details,
                               image_path=f"/uploads/{filename}",
                               retrained='0')

    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/retrain', methods=['POST'])
def retrain():
    global model

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
    model = load_model_from_file()

    return redirect(url_for('home', retrain_message="✅ Model retrained successfully.", retrained='1'))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
