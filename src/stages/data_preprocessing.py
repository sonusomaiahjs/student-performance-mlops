import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

class DataPreprocessing:
    def __init__(
        self, 
        validated_path="data/processed/validated_data.csv", 
        processed_dir="data/processed"
    ):
        self.validated_path = validated_path
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess(self, test_size=0.2, random_state=42):

        print(f"📥 Loading validated data from: {self.validated_path}")

        # Load validated data (NOT raw)
        df = pd.read_csv(self.validated_path)

        mlflow.set_experiment("student_performance")
        with mlflow.start_run(run_name="data_preprocessing"):

            mlflow.log_param("rows", df.shape[0])
            mlflow.log_param("columns", df.shape[1])

            # Features and target
            X = df.drop("score", axis=1)
            y = df["score"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Paths
            X_train_path = os.path.join(self.processed_dir, "X_train.csv")
            X_test_path  = os.path.join(self.processed_dir, "X_test.csv")
            y_train_path = os.path.join(self.processed_dir, "y_train.csv")
            y_test_path  = os.path.join(self.processed_dir, "y_test.csv")

            # Save outputs
            X_train.to_csv(X_train_path, index=False)
            X_test.to_csv(X_test_path, index=False)
            y_train.to_csv(y_train_path, index=False)
            y_test.to_csv(y_test_path, index=False)

            print("✅ Data Preprocessing Completed!")
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

            mlflow.log_metric("train_rows", len(X_train))
            mlflow.log_metric("test_rows", len(X_test))

            mlflow.log_artifact(X_train_path)
            mlflow.log_artifact(X_test_path)
            mlflow.log_artifact(y_train_path)
            mlflow.log_artifact(y_test_path)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.preprocess()
