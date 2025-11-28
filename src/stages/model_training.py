import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

class ModelTraining:
    def __init__(self, processed_dir="data/processed", model_dir="models"):
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "linear_regression_model.pkl")

    def train_model(self):
        # Load processed data
        X_train = pd.read_csv(os.path.join(self.processed_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(self.processed_dir, "y_train.csv"))
        X_test = pd.read_csv(os.path.join(self.processed_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(self.processed_dir, "y_test.csv"))

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("✅ Model Training Completed!")
        print(f"MSE: {mse:.2f}, R2 Score: {r2:.2f}")

        # Save model
        joblib.dump(model, self.model_path)
        print(f"💾 Model saved at: {self.model_path}")

        # MLflow logs
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)

        return model

if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.train_model()
