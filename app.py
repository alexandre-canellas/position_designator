from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

MODEL_PATH = "models"
MODEL = joblib.load(f"{MODEL_PATH}/{os.environ['model']}_model.pkl")


@app.route('/')
def home():
    return render_template("home.html"), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recebe um JSON com atributos de jogador da última temporada
    
    Exemplo:
      {
        'age': 27,
        'goals': 10,
        'assists': 4,
        'total_cards': 8,
        'min_per_match': 67
      }

    Retorna a posição do jogador em string
    """
    
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    return jsonify({
        "pred":  MODEL.predict(df)[0]
    })

if __name__ == "__main__":
    app.run(debug=True)
