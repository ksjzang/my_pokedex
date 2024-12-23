from flask import Flask, request, jsonify
from model_script import predict_pokemon

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file.save("temp.jpg")  # 업로드된 파일 저장

    try:
        label = predict_pokemon("temp.jpg")
        return jsonify({"label": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
