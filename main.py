import mlflow
import numpy as np
import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    load_data,
    predict,
)
from mlflow.tracking import MlflowClient
import os


# 🔹 Définir le backend de stockage en SQLite
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# client = MlflowClient()


def main():
    parser = argparse.ArgumentParser(
        description="Exécuter les différentes étapes du pipeline ML."
    )
    parser.add_argument(
        "step",
        type=str,
        choices=["prepare", "train", "evaluate", "save", "load", "predict"],
        help="Étape du pipeline à exécuter",
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Données d'entrée pour la prédiction (format CSV: '1.2,3.4,5.6,7.8')",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["Staging", "Production", "Archived"],
        default="Staging",
        help="Stage du modèle dans le Model Registry",
    )

    args = parser.parse_args()

    train_path = "churn-bigml-80.csv"
    test_path = "churn-bigml-20.csv"

    if args.step == "prepare":
        X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
        print("Données préparées avec succès.")
    elif args.step == "train":
        X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
        model = train_model(X_train, y_train)
        print("Modèle entraîné avec succès.")
    elif args.step == "evaluate":
        # Assure-toi que le modèle est entraîné avant d'évaluer
        # Si déjà entraîné, charge-le avec joblib.load() par ex.
        X_train, X_test, y_train, y_test = prepare_data(
            train_path, test_path
        )  # Adapter selon ton code
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
    elif args.step == "save":
        X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
        model = train_model(X_train, y_train)
        save_model(model)
        print(f"Modèle sauvegardé .")
    elif args.step == "load":
        model = load_model()
        print("Modèle chargé avec succès.")
    elif args.step == "predict":
        # Charger le modèle
        loaded_model = load_model()
        if args.features:
            features = np.array([list(map(float, args.features.split(",")))])
        else:
            _, X_test, _, _ = prepare_data(train_path, test_path)
            features = X_test[:1]  # Prendre un exemple

        with mlflow.start_run():
            mlflow.sklearn.log_model(loaded_model, "model")

            # 🔹 Ajouter le modèle au Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "model_predict")

            # 🔹 Récupérer la dernière version du modèle enregistré
            model_name = "model_predict"
            latest_version = client.get_latest_versions(model_name, stages=["None"])[
                0
            ].version

            # 🔹 Mettre à jour le statut du modèle en fonction de l'argument --stage
            stage = args.stage
            client.transition_model_version_stage(
                name=model_name, version=latest_version, stage=stage
            )

            print(f"\n✅ Model {model_name} version {latest_version} moved to {stage}!")

        prediction = predict(features)
        print(f"✅ Prédiction réalisée : {prediction}")


if __name__ == "__main__":
    # Démarrer MLflow avec un tracking URI local
    mlflow.set_tracking_uri("http://localhost:5001")
    # mlflow.set_tracking_uri("postgresql://rania:rania123@localhost/mlflow_db")
    mlflow.set_experiment("Churn_Prediction")
    client = MlflowClient()
    main()
