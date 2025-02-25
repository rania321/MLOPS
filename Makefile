# Définition des variables
PYTHON = python
VENV = venv
REQ = requirements.txt

# Chemins des fichiers
TRAIN_PATH = churn-bigml-80.csv
TEST_PATH = churn-bigml-20.csv

# Création de l'environnement virtuel et installation des dépendances
install:
	@echo "🔧 Création de l'environnement virtuel..."
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	@echo "📦 Installation des dépendances..."
	@$(VENV)/bin/pip install --upgrade pip
	@$(VENV)/bin/pip install -r $(REQ)
	@echo "✅ Installation terminée."

# Vérification et formatage du code
lint:
	@echo "🔍 Vérification du code..."
	@$(VENV)/bin/black .
	@$(VENV)/bin/pylint --fail-under=5 *.py
	@$(VENV)/bin/bandit -r . --exclude venv/
	@echo "✅ Code conforme."

# Préparer les données
prepare:
	@echo "📊 Préparation des données..."
	@$(VENV)/bin/python main.py prepare
	@echo "✅ Données préparées."


# Entraîner le modèle
train:
	@echo "🤖 Entraînement du modèle..."
	@$(VENV)/bin/python main.py train
	@echo "✅ Modèle entraîné."

# Évaluer le modèle
evaluate:
	@echo "📈 Évaluation du modèle..."
	@$(VENV)/bin/python main.py evaluate
	@echo "✅ Évaluation terminée."

# Sauvegarder le modèle
save:
	@echo "💾 Sauvegarde du modèle..."
	@$(VENV)/bin/python main.py save
	@echo "✅ Modèle sauvegardé."

# Charger le modèle
load:
	@echo "📂 Chargement du modèle..."
	@$(VENV)/bin/python main.py load
	@echo "✅ Modèle chargé."

# Nettoyage des fichiers temporaires
clean:
	@echo "🧹 Nettoyage des fichiers temporaires..."
	@rm -rf $(VENV) __pycache__ *.pkl *.joblib
	@echo "✅ Nettoyage terminé."

# 4. Tests unitaires (si un fichier test.py existe)
test:
	@echo "🧪 Exécution des tests..."
	pytest test_model.py

run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

retrain:
	curl -X POST "http://127.0.0.1:8000/retrain/" -H "Content-Type: application/json" -d '{"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3}'

predict:
	@$(VENV)/bin/python main.py predict --features "5.1, 3.5, 1.4, 0.2, 2, 5.2, 1.2, 1.4"

production:
	@$(VENV)/bin/python main.py predict --stage Production --features "5.1, 3.5, 1.4, 0.2, 2, 5.2, 1.2, 1.4"

archived:
	@$(VENV)/bin/python main.py predict --stage Archived --features "5.1, 3.5, 1.4, 0.2, 2, 5.2, 1.2, 1.4"

run-mlflow:
	@$(VENV)/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001

build:
	docker build -t fastapi-mlflow-app .

run:
	docker run -d -p 8000:8000 fastapi-mlflow-app

push:
	docker tag fastapi-mlflow-app raniaguelmami/fastapi-mlflow-app
	docker push raniaguelmami/fastapi-mlflow-app

run_docker:
	docker-compose up -d

stop_docker:
	docker-compose down

# Exécution de toutes les étapes
all: install lint prepare train evaluate save test



