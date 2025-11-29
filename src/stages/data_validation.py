import pandas as pd
import mlflow
import os

def validate_data(df: pd.DataFrame) -> dict:
    """
    Validates ingested data for:
    - missing values
    - correct column names
    - valid data types
    """
    report = {}

    required_columns = ["hours", "score"]
    report["has_required_columns"] = int(all(col in df.columns for col in required_columns))

    report["missing_values"] = int(df.isnull().sum().sum() > 0)
    report["hours_numeric"] = int(pd.api.types.is_numeric_dtype(df["hours"]))
    report["score_numeric"] = int(pd.api.types.is_numeric_dtype(df["score"]))

    all_ok = (
        report["has_required_columns"] == 1 and
        report["missing_values"] == 0 and
        report["hours_numeric"] == 1 and
        report["score_numeric"] == 1
    )

    print("Validation Summary:", report)
    return all_ok, report


def main():
    mlflow.set_experiment("student_performance")

    with mlflow.start_run(run_name="data_validation"):
        input_path = "data/processed/ingested_data.csv"
        df = pd.read_csv(input_path)

        mlflow.log_param("input_file", input_path)

        validation_status, report = validate_data(df)

        # Log metrics to MLflow
        for key, value in report.items():
            mlflow.log_metric(key, value)

        # Save validation report for DVC
        report_path = "data/processed/validation_report.csv"
        pd.DataFrame([report]).to_csv(report_path, index=False)

        print(f"📊 Validation report saved at: {report_path}")

        if validation_status:
            output_path = "data/processed/validated_data.csv"
            df.to_csv(output_path, index=False)
            print(f"📁 Validated data saved at: {output_path}")
        else:
            raise ValueError("❌ Data validation failed. Pipeline stopped.")


if __name__ == "__main__":
    main()
