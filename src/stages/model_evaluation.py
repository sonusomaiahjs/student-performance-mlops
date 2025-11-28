import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

class ModelEvaluation:
    def __init__(self, processed_dir="data/processed", model_path="models/linear_regression_model.pkl"):
        self.processed_dir = processed_dir
        self.model_path = model_path

    def evaluate(self):
        # Load model
        model = joblib.load(self.model_path)

        # Load test data
        X_test = pd.read_csv(os.path.join(self.processed_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(self.processed_dir, "y_test.csv"))

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("✅ Model Evaluation Completed!")
        print(f"MSE: {mse:.2f}, R2 Score: {r2:.2f}")
        print("\nActual vs Predicted:")
        comparison = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
        print(comparison)

        # MLflow logs
        mlflow.log_metric("Eval_MSE", mse)
        mlflow.log_metric("Eval_R2", r2)

        return comparison

if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.evaluate()
