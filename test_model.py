import pytest
import numpy as np
from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Test pour vérifier si la fonction prepare_data fonctionne correctement
def test_prepare_data():
    train_path = (
        "churn-bigml-80.csv"  # Assure-toi que ce fichier est dans ton répertoire
    )
    test_path = (
        "churn-bigml-20.csv"  # Assure-toi que ce fichier est dans ton répertoire
    )

    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)

    # Vérifie que les formes des données sont correctes
    assert X_train.shape[0] > 0, "X_train is empty"
    assert X_test.shape[0] > 0, "X_test is empty"
    assert y_train.shape[0] == X_train.shape[0], "Mismatch between X_train and y_train"
    assert y_test.shape[0] == X_test.shape[0], "Mismatch between X_test and y_test"


# Test pour vérifier si le modèle peut être correctement entraîné
def test_train_model():
    # On utilise des données fictives pour l'entraînement
    X_train = np.random.rand(100, 8)  # Exemple de 100 échantillons avec 8 features
    y_train = np.random.randint(0, 2, 100)  # 100 échantillons binaires (0 ou 1)

    model = train_model(X_train, y_train)

    # Vérifie que le modèle retourné est bien un GradientBoostingClassifier
    assert isinstance(
        model, GradientBoostingClassifier
    ), "The model is not a GradientBoostingClassifier"


# Test pour vérifier si la fonction evaluate_model retourne des résultats corrects
def test_evaluate_model():
    X_train = np.random.rand(100, 8)
    y_train = np.random.randint(0, 2, 100)
    model = train_model(X_train, y_train)

    X_test = np.random.rand(20, 8)
    y_test = np.random.randint(0, 2, 20)

    accuracy, precision, recall, f1, cm = evaluate_model(model, X_test, y_test)

    # Vérifie que les métriques sont bien des nombres
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(f1, float), "F1-score should be a float"


# Test pour vérifier si la fonction save_model et load_model fonctionnent correctement
def test_save_and_load_model():
    model = GradientBoostingClassifier()
    model.fit(np.random.rand(100, 8), np.random.randint(0, 2, 100))

    save_model(model, filename="test_model.joblib")
    loaded_model = load_model(filename="test_model.joblib")

    # Vérifie que le modèle chargé est bien un GradientBoostingClassifier
    assert isinstance(
        loaded_model, GradientBoostingClassifier
    ), "The loaded model is not a GradientBoostingClassifier"
