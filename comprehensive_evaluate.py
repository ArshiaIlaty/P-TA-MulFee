import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")


def load_data(train_file, test_file, label_column):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    y_train = train_df[label_column]
    y_test = test_df[label_column]

    X_train = train_df.drop(columns=[label_column])
    X_test = test_df.drop(columns=[label_column])

    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, X_test, y_test):
    encoders = {}
    valid_indices = X_test.index.tolist()

    for col in X_train.columns:
        if X_train[col].dtype == "object":
            encoders[col] = LabelEncoder()
            X_train[col] = encoders[col].fit_transform(X_train[col])

            # Handle unseen values in test set
            unseen_values = set(X_test[col]) - set(encoders[col].classes_)
            invalid_rows = X_test[X_test[col].isin(unseen_values)].index

            valid_indices = list(set(valid_indices) - set(invalid_rows))

            X_test = X_test.loc[valid_indices]
            y_test = y_test.loc[valid_indices]
            X_test[col] = encoders[col].transform(X_test[col])

    return X_train, X_test, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a single model and return metrics"""
    print(f"\n--- {model_name} ---")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def comprehensive_evaluation(train_file, test_file, label_column):
    print("=== COMPREHENSIVE SYNTHETIC DATA EVALUATION ===")
    print(f"Training on: {train_file}")
    print(f"Testing on: {test_file}")
    print(f"Target column: {label_column}")
    print()

    # Load data
    X_train, y_train, X_test, y_test = load_data(train_file, test_file, label_column)

    # Preprocess data
    X_train, X_test, y_test = preprocess_data(X_train, X_test, y_test)

    print(f"Final training set: {X_train.shape}")
    print(f"Final test set: {X_test.shape}")
    print()

    # Initialize models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    # Store results
    results = []

    # Evaluate each model
    for model_name, model in models.items():
        result = evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
        results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL MODELS")
    print("=" * 80)

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # Find best model
    best_model = summary_df.loc[summary_df["accuracy"].idxmax()]
    print(
        f"\nBest performing model: {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})"
    )

    return results, summary_df


if __name__ == "__main__":
    # Evaluate synthetic data quality with multiple models
    results, summary = comprehensive_evaluation(
        "output_heloc_clean.csv", "heloc.csv", "RiskPerformance"
    )
