import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder


def preprocess(X_train, X_test):
    encoders = {}
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            encoders[col] = LabelEncoder()
            X_train[col] = encoders[col].fit_transform(X_train[col])
            # Handle unseen values in test set
            X_test[col] = X_test[col].map(
                lambda s: (
                    s if s in encoders[col].classes_ else encoders[col].classes_[0]
                )
            )
            X_test[col] = encoders[col].transform(X_test[col])
    return X_train, X_test


def evaluate(train_file, test_file, label_column, desc):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    X_train, y_train = train_df.drop(columns=[label_column]), train_df[label_column]
    X_test, y_test = test_df.drop(columns=[label_column]), test_df[label_column]
    X_train, X_test = preprocess(X_train, X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n--- {desc} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Train on real, test on real
    evaluate("diabetes.csv", "diabetes.csv", "diabetes", "Train on Real, Test on Real")
    # Train on synthetic, test on real
    evaluate(
        "output_diabetes_clean.csv",
        "diabetes.csv",
        "diabetes",
        "Train on Synthetic, Test on Real",
    )
    # Optionally, train on real, test on synthetic
    evaluate(
        "diabetes.csv",
        "output_diabetes_clean.csv",
        "diabetes",
        "Train on Real, Test on Synthetic",
    )
