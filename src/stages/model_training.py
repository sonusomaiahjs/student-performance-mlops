import os
import pandas as pd
import mlflow
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class ModelTraining:
    def __init__(self, processed_dir="data/processed", model_dir="models"):
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self):
        print("📥 Loading training data...")

        X_train = pd.read_csv(os.path.join(self.processed_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(self.processed_dir, "y_train.csv"))

        # Convert y_train to Series (important)
        y_train = y_train.squeeze()

        mlflow.set_experiment("student_performance")

        with mlflow.start_run():
            mlflow.log_param("model_type", "LinearRegression")

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions for metrics
            preds = model.predict(X_train)

            mse = mean_squared_error(y_train, preds)
            r2 = r2_score(y_train, preds)

            mlflow.log_metric("train_mse", mse)
            mlflow.log_metric("train_r2", r2)

            print(f"📊 Training MSE: {mse}")
            print(f"📈 Training R²: {r2}")

            # Save model locally
            model_path = os.path.join(self.model_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(model_path)
            print(f"✅ Model saved at: {model_path}")


if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.train()
