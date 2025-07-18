import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def load_and_preprocess_data(file_path, label_column):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)

    # Clean synthetic data if needed
    if "output" in file_path:
        df_clean = df.copy()
        for col in df_clean.columns:
            if col != label_column:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean.dropna()
        df = df_clean

    y = df[label_column]
    X = df.drop(columns=[label_column])

    # Handle categorical columns
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    return X, y


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a single model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compare_real_vs_synthetic():
    print("=== REAL vs SYNTHETIC DATA COMPARISON ===")
    print("Evaluating model performance on both datasets")
    print()

    # Load real data
    print("Loading real data...")
    X_real, y_real = load_and_preprocess_data("heloc.csv", "RiskPerformance")

    # Load synthetic data
    print("Loading synthetic data...")
    X_synthetic, y_synthetic = load_and_preprocess_data(
        "output_heloc_clean.csv", "RiskPerformance"
    )

    print(f"Real data shape: {X_real.shape}")
    print(f"Synthetic data shape: {X_synthetic.shape}")
    print()

    # Split real data for training/testing
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
    )

    # Use synthetic data for training, real data for testing
    X_synthetic_train, y_synthetic_train = X_synthetic, y_synthetic
    X_synthetic_test, y_synthetic_test = X_real_test, y_real_test

    # Initialize models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    # Store results
    results = []

    # Evaluate on real data (baseline)
    print("--- EVALUATION ON REAL DATA (Baseline) ---")
    for model_name, model in models.items():
        result = evaluate_model(
            model, X_real_train, y_real_train, X_real_test, y_real_test, model_name
        )
        result["data_type"] = "Real"
        results.append(result)
        print(f"{model_name}: Accuracy = {result['accuracy']:.4f}")

    print()

    # Evaluate on synthetic data
    print("--- EVALUATION ON SYNTHETIC DATA ---")
    for model_name, model in models.items():
        result = evaluate_model(
            model,
            X_synthetic_train,
            y_synthetic_train,
            X_synthetic_test,
            y_synthetic_test,
            model_name,
        )
        result["data_type"] = "Synthetic"
        results.append(result)
        print(f"{model_name}: Accuracy = {result['accuracy']:.4f}")

    # Create comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: REAL vs SYNTHETIC DATA")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    # Pivot for better comparison
    comparison_df = results_df.pivot(
        index="model",
        columns="data_type",
        values=["accuracy", "precision", "recall", "f1_score"],
    )
    comparison_df.columns = [f"{col[1]}_{col[0]}" for col in comparison_df.columns]

    print(comparison_df.round(4))

    # Calculate utility preservation
    print("\n" + "=" * 80)
    print("UTILITY PRESERVATION ANALYSIS")
    print("=" * 80)

    for model_name in models.keys():
        real_acc = results_df[
            (results_df["model"] == model_name) & (results_df["data_type"] == "Real")
        ]["accuracy"].iloc[0]
        synth_acc = results_df[
            (results_df["model"] == model_name)
            & (results_df["data_type"] == "Synthetic")
        ]["accuracy"].iloc[0]
        preservation = (synth_acc / real_acc) * 100

        print(f"{model_name}:")
        print(f"  Real Data Accuracy: {real_acc:.4f}")
        print(f"  Synthetic Data Accuracy: {synth_acc:.4f}")
        print(f"  Utility Preservation: {preservation:.2f}%")
        print()

    return results_df, comparison_df


if __name__ == "__main__":
    results, comparison = compare_real_vs_synthetic()
