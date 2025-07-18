import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
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
    """Load and prepare data for training and testing"""
    print(f"Loading training data from: {train_file}")
    print(f"Loading test data from: {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Separate features and target
    y_train = train_df[label_column]
    y_test = test_df[label_column]

    X_train = train_df.drop(columns=[label_column])
    X_test = test_df.drop(columns=[label_column])

    print(f"Final training set: {X_train.shape}")
    print(f"Final test set: {X_test.shape}")
    print()

    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, X_test, y_test):
    """Preprocess categorical variables and handle unseen values"""
    encoders = {}
    valid_indices = X_test.index.tolist()

    for col in X_train.columns:
        if X_train[col].dtype == "object":
            encoders[col] = LabelEncoder()
            X_train[col] = encoders[col].fit_transform(X_train[col])

            # Handle unseen values in test set
            unseen_values = set(X_test[col]) - set(encoders[col].classes_)
            if unseen_values:
                invalid_rows = X_test[X_test[col].isin(unseen_values)].index
                valid_indices = list(set(valid_indices) - set(invalid_rows))

                # Update test data
                X_test = X_test.loc[valid_indices]
                y_test = y_test.loc[valid_indices]

            X_test[col] = encoders[col].transform(X_test[col])

    return X_train, X_test, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train and evaluate a single model"""
    print(f"--- {model_name} ---")

    # Train the model
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
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():
    print("=== COMPREHENSIVE DIABETES SYNTHETIC DATA EVALUATION ===")
    print("Training on: output_diabetes_clean.csv")
    print("Testing on: diabetes.csv")
    print("Target column: diabetes")
    print()

    # Load data
    X_train, y_train, X_test, y_test = load_data(
        "output_diabetes_clean.csv", "diabetes.csv", "diabetes"
    )

    # Preprocess data
    X_train, X_test, y_test = preprocess_data(X_train, X_test, y_test)

    # Define models
    models = [
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (LogisticRegression(random_state=42, max_iter=1000), "Logistic Regression"),
    ]

    # Evaluate each model
    results = []
    for model, name in models:
        result = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results.append(result)

    # Create summary
    print("=" * 80)
    print("SUMMARY OF ALL MODELS")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()

    # Find best model
    best_model = results_df.loc[results_df["accuracy"].idxmax()]
    print(
        f"Best performing model: {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})"
    )

    # Additional analysis
    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS")
    print("=" * 80)

    # Compare with real data performance
    print("\n--- Real vs Synthetic Data Comparison ---")

    # Load original diabetes data
    original_df = pd.read_csv("diabetes.csv")

    # Split original data for comparison
    from sklearn.model_selection import train_test_split

    X_orig = original_df.drop(columns=["diabetes"])
    y_orig = original_df["diabetes"]

    # Preprocess original data
    X_orig_processed, _, _ = preprocess_data(X_orig, X_orig, y_orig)

    # Train on real data, test on same real data
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_orig_processed, y_orig, test_size=0.3, random_state=42
    )

    # Compare Random Forest performance
    rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_real.fit(X_train_real, y_train_real)
    y_pred_real = rf_real.predict(X_test_real)
    real_accuracy = accuracy_score(y_test_real, y_pred_real)

    print(f"Random Forest on Real Data: {real_accuracy:.4f}")
    print(f"Random Forest on Synthetic Data: {best_model['accuracy']:.4f}")
    print(f"Utility Preservation: {best_model['accuracy']/real_accuracy*100:.2f}%")

    # Data quality metrics
    print("\n--- Data Quality Metrics ---")
    synthetic_df = pd.read_csv("output_diabetes_clean.csv")
    original_df = pd.read_csv("diabetes.csv")

    print(f"Original dataset size: {len(original_df)}")
    print(f"Synthetic dataset size: {len(synthetic_df)}")
    print(f"Data retention after cleaning: {len(synthetic_df)/100000*100:.1f}%")

    # Feature distribution comparison
    print("\n--- Feature Distribution Comparison ---")
    numerical_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

    for feature in numerical_features:
        orig_mean = original_df[feature].mean()
        synth_mean = synthetic_df[feature].mean()
        orig_std = original_df[feature].std()
        synth_std = synthetic_df[feature].std()

        print(f"{feature}:")
        print(f"  Original - Mean: {orig_mean:.2f}, Std: {orig_std:.2f}")
        print(f"  Synthetic - Mean: {synth_mean:.2f}, Std: {synth_std:.2f}")
        print(f"  Mean difference: {abs(orig_mean - synth_mean):.2f}")
        print()


if __name__ == "__main__":
    main()
