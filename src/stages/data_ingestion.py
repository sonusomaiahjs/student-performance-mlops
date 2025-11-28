import os
import pandas as pd
import mlflow

class DataIngestion:
    def __init__(self, input_path="data/raw/student_scores.csv"):
        self.input_path = os.path.abspath(input_path)

    def load_data(self):
        # ✅ All lines inside the function must be indented
        print(f"🔎 Looking for file at: {self.input_path}")
        mlflow.log_param("raw_data_path", self.input_path)

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        mlflow.log_metric("raw_rows", df.shape[0])
        mlflow.log_metric("raw_columns", df.shape[1])
        print("📥 Data Ingestion Completed!")

        # Save processed file
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        output_path = os.path.join(processed_dir, "ingested_data.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Ingested data saved at: {output_path}")

        return df

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    print(df.head())
