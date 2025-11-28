import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

class DataPreprocessing:
    def __init__(self, raw_path="data/raw/student_scores.csv", processed_dir="data/processed"):
        self.raw_path = raw_path
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess(self, test_size=0.2, random_state=42):
        # Load raw data
        df = pd.read_csv(self.raw_path)
        mlflow.log_param("raw_rows", df.shape[0])
        mlflow.log_param("raw_columns", df.shape[1])

        # Split features and target
        X = df.drop("score", axis=1)
        y = df["score"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Save processed data
        X_train.to_csv(os.path.join(self.processed_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.processed_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.processed_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.processed_dir, "y_test.csv"), index=False)

        print("✅ Data Preprocessing Completed!")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))

        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.preprocess()
