import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


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


def train_and_evaluate(train_file, test_file, label_column):
    print("=== HELOC Synthetic Data Evaluation ===")
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

    # Train model
    print("Training Decision Tree classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("=== Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print()

    # Detailed classification report
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    # Evaluate synthetic data quality
    train_and_evaluate("output_heloc_clean.csv", "heloc.csv", "RiskPerformance")
