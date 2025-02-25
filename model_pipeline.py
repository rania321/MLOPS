import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
    precision_score,
)
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

client = MlflowClient()  # Client pour gérer le Model Registry


def load_data(train_path="churn-bigml-80.csv", test_path="churn-bigml-20.csv"):
    """
    Charge les fichiers de données et effectue la séparation des features et labels.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Adapter selon tes colonnes
    X_train = df_train.drop(
        columns=["target"]
    )  # Remplace "target" par le nom réel de la colonne cible
    y_train = df_train["target"]

    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    return X_train, X_test, y_train, y_test


def prepare_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_prep = df.copy()

    df_prep["International plan"] = df_prep["International plan"].map(
        {"Yes": 1, "No": 0}
    )
    df_prep["Voice mail plan"] = df_prep["Voice mail plan"].map({"Yes": 1, "No": 0})
    df_prep["Churn"] = df_prep["Churn"].astype(int)

    target_mean = df_prep.groupby("State")["Churn"].mean()
    df_prep["STATE_TargetMean"] = df_prep["State"].map(target_mean)

    label_encoder = LabelEncoder()
    df_prep["STATE_Label"] = label_encoder.fit_transform(df_prep["State"])
    df_prep = df_prep.drop(columns=["State"])

    corr_data = df_prep.corr()
    upper_triangle = corr_data.where(
        np.triu(np.ones(corr_data.shape), k=1).astype(bool)
    )
    high_correlation_columns = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)
    ]
    df_prep_dropped = df_prep.drop(columns=high_correlation_columns)

    lower_limit = df_prep_dropped.quantile(0.05)
    upper_limit = df_prep_dropped.quantile(0.95)
    df_prep_clipped = df_prep_dropped.apply(
        lambda x: x.clip(lower_limit[x.name], upper_limit[x.name])
    )

    X = df_prep_clipped.drop(columns=["Churn"])
    y = df_prep_clipped["Churn"]

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    columns_to_drop = [
        "STATE_TargetMean",
        "STATE_Label",
        "Account length",
        "Total night calls",
        "Area code",
        "Total day calls",
        "Total eve calls",
    ]
    X = X.drop(columns=columns_to_drop, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):
    param_dist = {
        "n_estimators": randint(50, 200),
        "learning_rate": uniform(0.01, 0.2),
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    }

    gb_model = GradientBoostingClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        gb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )

    with mlflow.start_run():
        # Suivi de l'accuracy pendant l'entraînement
        accuracies = []  # Liste pour stocker l'accuracy à chaque itération
        for i in range(5):  # Nombre d'itérations (modifie selon ton nombre d'itérations)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_

            # Calcul de l'accuracy après chaque itération et ajout à la liste
            accuracy = best_model.score(X_train, y_train)
            accuracies.append(accuracy)

            # Log des métriques
            mlflow.log_metric("train_accuracy", accuracy)
        
        # Enregistrer la courbe de précision pendant l'entraînement
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(accuracies) + 1), accuracies, label="Training Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy Curve")
        plt.legend()

        # Sauvegarder la figure dans MLflow
        mlflow.log_figure(plt.gcf(), "training_accuracy_curve.png")
        plt.close()  # Pour fermer la figure après enregistrement

        # Log des hyperparamètres
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("learning_rate", best_model.learning_rate)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("subsample", best_model.subsample)

        # Enregistrement du modèle
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="models",
            registered_model_name="GradientBoostingModel",
        )

    return best_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

    # Affichage des résultats
    print("📊 **Évaluation du modèle** 📊")
    print(f"🔹 Accuracy : {accuracy:.4f}")
    print(f"🔹 Precision : {precision:.4f}")
    print(f"🔹 Recall : {recall:.4f}")
    print(f"🔹 F1-Score : {f1:.4f}")

    return accuracy, precision, recall, f1


def save_model(model, filename="model.joblib"):
    joblib.dump(model, filename)
    try:
        mlflow.log_artifact(filename)
        print(f"✅ Artefact enregistré : {filename}")
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement de l'artefact : {e}")


def load_model(filename="model.joblib"):
    return joblib.load(filename)


def predict(features):
    """
    Prédiction avec un modèle pré-entraîné et enregistrement des résultats dans MLflow et Elasticsearch.
    """
    model = joblib.load("model.joblib")
    prediction = model.predict(features)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("features", features.tolist())
        mlflow.log_param("prediction", prediction.tolist())
        mlflow.log_metric("prediction", prediction[0])
        # Enregistrement du modèle

    print(f"✅ Prédiction terminée ! Prediction: {prediction}")
    return prediction
