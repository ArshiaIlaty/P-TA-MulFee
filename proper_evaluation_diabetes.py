import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")


def analyze_class_distribution(df, dataset_name):
    """Analyze and display class distribution"""
    print(f"\n=== {dataset_name} Class Distribution ===")
    print(f"Total samples: {len(df)}")

    class_counts = df["diabetes"].value_counts()
    class_percentages = df["diabetes"].value_counts(normalize=True) * 100

    print(
        f"Class 0 (No Diabetes): {class_counts[0]} samples ({class_percentages[0]:.2f}%)"
    )
    print(
        f"Class 1 (Diabetes): {class_counts[1]} samples ({class_percentages[1]:.2f}%)"
    )
    print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")

    return class_counts, class_percentages


def apply_balancing(X, y, method):
    if method == "none":
        return X, y
    elif method == "undersample":
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        return X_res, y_res
    elif method == "oversample":
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        return X_res, y_res
    elif method == "smote":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    else:
        raise ValueError(f"Unknown balancing method: {method}")


def evaluate_with_stratified_cv(
    X, y, model, cv_folds=5, model_name="Model", balancing_method="none"
):
    """Evaluate model with stratified cross-validation"""
    print(f"\n=== {model_name} - Stratified Cross-Validation ({balancing_method}) ===")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Store metrics for each fold
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "precision_class_0": [],
        "recall_class_0": [],
        "f1_class_0": [],
        "precision_class_1": [],
        "recall_class_1": [],
        "f1_class_1": [],
        "auc": [],
        "avg_precision": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply balancing to training data only
        X_train_bal, y_train_bal = apply_balancing(X_train, y_train, balancing_method)

        # Train model
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Class-specific metrics
        precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

        precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        # AUC and Average Precision
        auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        # Store metrics
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1"].append(f1)
        fold_metrics["precision_class_0"].append(precision_0)
        fold_metrics["recall_class_0"].append(recall_0)
        fold_metrics["f1_class_0"].append(f1_0)
        fold_metrics["precision_class_1"].append(precision_1)
        fold_metrics["recall_class_1"].append(recall_1)
        fold_metrics["f1_class_1"].append(f1_1)
        fold_metrics["auc"].append(auc)
        fold_metrics["avg_precision"].append(avg_precision)

        print(f"Fold {fold}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        print(f"  Class 0: P={precision_0:.4f}, R={recall_0:.4f}, F1={f1_0:.4f}")
        print(f"  Class 1: P={precision_1:.4f}, R={recall_1:.4f}, F1={f1_1:.4f}")

    # Calculate mean and std
    print(f"\n{model_name} - Cross-Validation Summary:")
    print(
        f"Accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}"
    )
    print(
        f"F1-Score: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}"
    )
    print(
        f"AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}"
    )
    print(
        f"Average Precision: {np.mean(fold_metrics['avg_precision']):.4f} ± {np.std(fold_metrics['avg_precision']):.4f}"
    )

    return fold_metrics


def evaluate_synthetic_vs_real(synthetic_csv, real_csv):
    """Compare synthetic vs real data performance"""
    print("=== SYNTHETIC VS REAL DATA EVALUATION ===")

    # Load data
    synthetic_df = pd.read_csv(synthetic_csv)
    real_df = pd.read_csv(real_csv)

    # Analyze class distributions
    real_counts, real_percentages = analyze_class_distribution(
        real_df, "Original Dataset"
    )
    synth_counts, synth_percentages = analyze_class_distribution(
        synthetic_df, "Synthetic Dataset"
    )

    # Prepare features and target
    feature_cols = [col for col in real_df.columns if col != "diabetes"]

    # Handle categorical variables
    from sklearn.preprocessing import LabelEncoder

    # Combine datasets for consistent encoding
    combined_df = pd.concat(
        [real_df[feature_cols], synthetic_df[feature_cols]], ignore_index=True
    )

    # Encode categorical variables
    encoders = {}
    X_encoded = combined_df.copy()

    for col in feature_cols:
        if combined_df[col].dtype == "object":
            encoders[col] = LabelEncoder()
            X_encoded[col] = encoders[col].fit_transform(combined_df[col])

    # Split back to real and synthetic
    X_real = X_encoded.iloc[: len(real_df)]
    X_synth = X_encoded.iloc[len(real_df) :]
    y_real = real_df["diabetes"]
    y_synth = synthetic_df["diabetes"]

    # Initialize models
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        ),
    }

    balancing_methods = ["none", "undersample", "oversample", "smote"]
    results = {}

    # Evaluate on real data
    print("\n" + "=" * 60)
    print("EVALUATION ON REAL DATA")
    print("=" * 60)

    for balancing_method in balancing_methods:
        print(f"\n{'='*60}\nEVALUATION ON REAL DATA ({balancing_method})\n{'='*60}")
        for name, model in models.items():
            print(f"\n{name} on Real Data:")
            results[f"{name}_real_{balancing_method}"] = evaluate_with_stratified_cv(
                X_real,
                y_real,
                model,
                model_name=f"{name} (Real)",
                balancing_method=balancing_method,
            )

    # Evaluate on synthetic data
    print("\n" + "=" * 60)
    print("EVALUATION ON SYNTHETIC DATA")
    print("=" * 60)

    for balancing_method in balancing_methods:
        print(
            f"\n{'='*60}\nEVALUATION ON SYNTHETIC DATA ({balancing_method})\n{'='*60}"
        )
        for name, model in models.items():
            print(f"\n{name} on Synthetic Data:")
            results[f"{name}_synthetic_{balancing_method}"] = (
                evaluate_with_stratified_cv(
                    X_synth,
                    y_synth,
                    model,
                    model_name=f"{name} (Synthetic)",
                    balancing_method=balancing_method,
                )
            )

    # Train on synthetic, test on real
    print("\n" + "=" * 60)
    print("TRAIN ON SYNTHETIC, TEST ON REAL")
    print("=" * 60)

    for balancing_method in balancing_methods:
        print(
            f"\n{'='*60}\nTRAIN ON SYNTHETIC, TEST ON REAL ({balancing_method})\n{'='*60}"
        )
        for name, model in models.items():
            print(f"\n{name} - Train on Synthetic, Test on Real:")

            # Use stratified split for synthetic training data
            X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
                X_synth, y_synth, test_size=0.2, stratify=y_synth, random_state=42
            )

            # Apply balancing to synthetic training data
            X_synth_train_bal, y_synth_train_bal = apply_balancing(
                X_synth_train, y_synth_train, balancing_method
            )

            # Train on synthetic
            model.fit(X_synth_train_bal, y_synth_train_bal)

            # Test on real data
            y_pred = model.predict(X_real)
            y_pred_proba = model.predict_proba(X_real)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_real, y_pred)
            precision = precision_score(y_real, y_pred, average="weighted")
            recall = recall_score(y_real, y_pred, average="weighted")
            f1 = f1_score(y_real, y_pred, average="weighted")

            # Class-specific metrics
            precision_0 = precision_score(y_real, y_pred, pos_label=0, zero_division=0)
            recall_0 = recall_score(y_real, y_pred, pos_label=0, zero_division=0)
            f1_0 = f1_score(y_real, y_pred, pos_label=0, zero_division=0)

            precision_1 = precision_score(y_real, y_pred, pos_label=1, zero_division=0)
            recall_1 = recall_score(y_real, y_pred, pos_label=1, zero_division=0)
            f1_1 = f1_score(y_real, y_pred, pos_label=1, zero_division=0)

            auc = roc_auc_score(y_real, y_pred_proba)
            avg_precision = average_precision_score(y_real, y_pred_proba)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
            print(
                f"Class 0 (No Diabetes): P={precision_0:.4f}, R={recall_0:.4f}, F1={f1_0:.4f}"
            )
            print(
                f"Class 1 (Diabetes): P={precision_1:.4f}, R={recall_1:.4f}, F1={f1_1:.4f}"
            )

            # Print classification report
            print("\nClassification Report:")
            print(
                classification_report(
                    y_real, y_pred, target_names=["No Diabetes", "Diabetes"]
                )
            )

            results[f"{name}_synthetic_to_real_{balancing_method}"] = {
                "accuracy": accuracy,
                "f1": f1,
                "auc": auc,
                "avg_precision": avg_precision,
                "precision_0": precision_0,
                "recall_0": recall_0,
                "f1_0": f1_0,
                "precision_1": precision_1,
                "recall_1": recall_1,
                "f1_1": f1_1,
            }

    return results


def create_evaluation_summary(results):
    """Create a summary table of all results"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)

    summary_data = []

    for key, metrics in results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            acc = metrics["accuracy"]
            f1 = metrics["f1"]
            auc = metrics["auc"]
            f1_0 = metrics.get("f1_0", None)
            f1_1 = metrics.get("f1_1", None)

            # If list, print mean ± std; if float, print value
            def fmt(val):
                if isinstance(val, list):
                    return f"{np.mean(val):.4f} ± {np.std(val):.4f}"
                else:
                    return f"{val:.4f}"

            summary_data.append(
                {
                    "Model": key.replace("_synthetic_to_real", "")
                    .replace("_real", " (Real)")
                    .replace("_synthetic", " (Synthetic)"),
                    "Training Data": (
                        "Synthetic" if "_synthetic_to_real" in key else "Same as Test"
                    ),
                    "Test Data": (
                        "Real" if "_synthetic_to_real" in key else "Same as Train"
                    ),
                    "Accuracy": fmt(acc),
                    "F1-Score": fmt(f1),
                    "AUC": fmt(auc),
                    "F1-Class-0": fmt(f1_0) if f1_0 is not None else "",
                    "F1-Class-1": fmt(f1_1) if f1_1 is not None else "",
                }
            )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print(
        "\nNOTE: All classifiers use class_weight='balanced' to handle class imbalance."
    )
    return summary_df


if __name__ == "__main__":
    # Use class_weight='balanced' for all classifiers
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Patch the models in the script to use class_weight='balanced'
    def get_models():
        return {
            "Decision Tree": DecisionTreeClassifier(
                random_state=42, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            ),
            "Logistic Regression": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
        }

    # Monkey-patch models in evaluate_synthetic_vs_real
    import builtins

    builtins.get_models = get_models

    # Run comprehensive evaluation
    results = evaluate_synthetic_vs_real(
        "output_hierarchical_clean.csv", "diabetes.csv"
    )

    # Create summary
    summary_df = create_evaluation_summary(results)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print("Key findings:")
    print(
        "1. Class imbalance properly handled with stratified sampling and class weights"
    )
    print("2. Cross-validation provides reliable performance estimates")
    print("3. Class-specific metrics show true model performance")
    print("4. Synthetic-to-real evaluation tests utility preservation")
