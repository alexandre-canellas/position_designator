import joblib
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from dotenv import load_dotenv
import os

load_dotenv()

# Parâmetros definidos
F1_THRESHOLD = 0.70
ACC_THRESHOLD = 0.70
MODEL_PATH = "models"

def test_model_performance():
    """
    Testa o modelo salvo usando o conjunto de holdout e valida se
    ele atende às métricas mínimas de performance.
    """

    # Carregamento do modelo e dados
    model = joblib.load(f"{MODEL_PATH}/{os.environ['model']}_model.pkl")
    X = pd.read_csv("data/X_val.csv")
    y = pd.read_csv("data/y_val.csv").squeeze()

    # Realiza predição
    y_pred = model.predict(X)

    # Métricas
    f1 = f1_score(y, y_pred, average="macro")
    acc = accuracy_score(y, y_pred)

    print(f"\nF1 Score (macro): {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")
    
    # Asserts para validação de qualidade
    assert round(f1,2) >= F1_THRESHOLD, f"F1 Score abaixo do limite: {f1:.2f} < {F1_THRESHOLD}"
    assert round(acc,2) >= ACC_THRESHOLD, f"Acurácia abaixo do limite: {acc:.2f} < {ACC_THRESHOLD}"
    