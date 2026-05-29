from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# =========================
# DNA MAPPING
# =========================

mapping = {
    'A': [1,0,0,0],
    'T': [0,1,0,0],
    'C': [0,0,1,0],
    'G': [0,0,0,1]
}

# =========================
# ENCODING FUNCTION
# =========================

def encode_sequence(seq, max_len=100):

    seq = seq.upper()

    # Padding
    if len(seq) < max_len:
        seq = seq + "A" * (max_len - len(seq))

    # Truncating
    seq = seq[:max_len]

    encoded = []

    for nuc in seq:
        encoded.append(mapping.get(nuc, [0,0,0,0]))

    return np.array(encoded)

# =========================
# PREDICTION FUNCTION
# =========================

def predict_disease(sequence):

    encoded = encode_sequence(sequence)

    # CORRECT SHAPE
    encoded = encoded.reshape(1, 100, 4)

    prediction = model.predict(encoded)

    probability = float(prediction[0][0])

    if probability > 0.5:
        return f"Disease Detected ({probability:.2f})"

    else:
        return f"Healthy ({1 - probability:.2f})"

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET", "POST"])

def home():

    result = ""

    if request.method == "POST":

        sequence = request.form["sequence"]

        valid_chars = set("ATCG")

        if not set(sequence.upper()).issubset(valid_chars):

            result = "Invalid DNA Sequence"

        else:

            result = predict_disease(sequence)

    return render_template("index.html", result=result)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)