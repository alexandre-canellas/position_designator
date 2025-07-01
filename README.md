# Technico

Projeto de ML desenvolvido para classificar a melhor posição de um jogador conforme sua performance.
Projeto open source com teste integrado com threshold de f1-score!

---

## 🗂️ Arquitetura do Repositório

```
├── app.py                                      # Flask app com carregamento do modelo e rota de predição
├── models/
│ ├── tree_model.pkl                            # Modelo pre treinado de Árvore de Decisão (scikit-learn)
│ └── knn_model.pkl                             # Modelo pre treinado de KNN (scikit-learn)
├── data/
│ ├── X_train.csv                               # Holdout features
│ └── y_train.csv                               # Holdout target
├── notebook/
│ └── modelagem.ipynb                           # Notebook utilizado na criação do modelo de ML
├── templates/
│ └── index.html                                # Frontend com formulário (HTML)
├── static/
│ └── style.css                                 # CSS do frontend
├── tests/
│ └── test_model_performance.py                 # PyTest com F1 score threshold
├── requirements.txt
└── README.md
```

---

## 🚀 Run local

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute o app

```bash
python app.py
```

---

## 🧪 Testes Automatizados

### Threshold definido
F1 Score (macro): ≥ 0.70

### Executando o teste

```bash
pytest tests/test_model_performance.py -v
```

## 📌 Requisitos minimos

Crie um arquivo .env na pasta raíz do repositório com uma variáve model:

```bash
model = 'tree'
```

O modelo 'tree' é o modelo que atende aos requisitos do teste (f1-score >= 0.7).
Caso queira simular um modelo que falhe nos testes, basta substituir o valor por 'knn'
