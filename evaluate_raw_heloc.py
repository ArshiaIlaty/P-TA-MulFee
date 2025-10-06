import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_raw_synthetic_data(real_csv="heloc.csv", synthetic_csv="output_hierarchical_heloc.csv"):
    """
    Evaluate raw synthetic HELOC data without any cleaning or manipulation
    """
    logger.info("=== EVALUATING RAW SYNTHETIC HELOC DATA ===")
    
    # Load data
    df_real = pd.read_csv(real_csv)
    df_synthetic = pd.read_csv(synthetic_csv)
    
    logger.info(f"Real HELOC data: {len(df_real)} samples")
    logger.info(f"Raw synthetic HELOC data: {len(df_synthetic)} samples")
    
    # Data quality assessment
    logger.info("=== RAW DATA QUALITY ASSESSMENT ===")
    
    # Check for missing values
    real_missing = df_real.isnull().sum().sum()
    synthetic_missing = df_synthetic.isnull().sum().sum()
    logger.info(f"Missing values - Real: {real_missing}, Synthetic: {synthetic_missing}")
    
    # Check for -9 values (missing value indicators)
    synthetic_minus_nine = (df_synthetic == -9).sum().sum()
    logger.info(f"-9 values in synthetic data: {synthetic_minus_nine}")
    
    # Count rows with all -9 values
    all_minus_nine_rows = (df_synthetic == -9).all(axis=1).sum()
    logger.info(f"Rows with all -9 values: {all_minus_nine_rows}")
    
    # Count rows with any -9 values
    any_minus_nine_rows = (df_synthetic == -9).any(axis=1).sum()
    logger.info(f"Rows with any -9 values: {any_minus_nine_rows}")
    
    # Target distribution
    logger.info("Target distribution - Real data:")
    print(df_real["RiskPerformance"].value_counts())
    
    logger.info("Target distribution - Synthetic data:")
    print(df_synthetic["RiskPerformance"].value_counts())
    
    # Prepare data for evaluation
    X_real, y_real = df_real.drop("RiskPerformance", axis=1), df_real["RiskPerformance"]
    X_synthetic, y_synthetic = (
        df_synthetic.drop("RiskPerformance", axis=1),
        df_synthetic["RiskPerformance"],
    )
    
    # Split real data
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    
    # Identify valid synthetic data (no missing values, no -9 values)
    logger.info("=== IDENTIFYING VALID SYNTHETIC DATA ===")
    
    missing_mask = X_synthetic.isnull().any(axis=1)
    minus_nine_mask = (X_synthetic == -9).any(axis=1)
    invalid_mask = missing_mask | minus_nine_mask
    
    valid_count = (~invalid_mask).sum()
    invalid_count = invalid_mask.sum()
    
    logger.info(f"Valid synthetic rows (no missing/-9): {valid_count}")
    logger.info(f"Invalid synthetic rows (with missing/-9): {invalid_count}")
    logger.info(f"Valid percentage: {valid_count/len(X_synthetic)*100:.1f}%")
    
    if valid_count == 0:
        logger.error("No valid synthetic data for evaluation!")
        return
    
    # Use only valid synthetic data
    X_synthetic_valid = X_synthetic[~invalid_mask]
    y_synthetic_valid = y_synthetic[~invalid_mask]
    
    logger.info(f"Using {len(X_synthetic_valid)} valid synthetic samples for evaluation")
    
    # Train models on synthetic data and test on real data
    logger.info("=== UTILITY PRESERVATION EVALUATION ===")
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name} on valid synthetic data...")
        
        try:
            # Train on synthetic data
            model.fit(X_synthetic_valid, y_synthetic_valid)
            
            # Predict on real test data
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            
            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
            
            logger.info(
                f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            results[name] = None
    
    # Baseline: Train on real data
    logger.info("=== BASELINE EVALUATION (Real Data) ===")
    baseline_results = {}
    
    for name, model in models.items():
        try:
            # Train on real training data
            model.fit(X_train, y_train)
            
            # Predict on real test data
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            baseline_results[name] = accuracy
            
            logger.info(f"{name} Baseline - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error training baseline {name}: {str(e)}")
            baseline_results[name] = None
    
    # Calculate utility preservation
    logger.info("=== UTILITY PRESERVATION RESULTS ===")
    for name in results.keys():
        if results[name] is not None and baseline_results[name] is not None:
            synthetic_acc = results[name]["accuracy"]
            baseline_acc = baseline_results[name]
            utility_preservation = (synthetic_acc / baseline_acc) * 100
            
            logger.info(f"{name}:")
            logger.info(f"  Synthetic Accuracy: {synthetic_acc:.4f}")
            logger.info(f"  Baseline Accuracy: {baseline_acc:.4f}")
            logger.info(f"  Utility Preservation: {utility_preservation:.2f}%")
            logger.info(f"  Precision: {results[name]['precision']:.4f}")
            logger.info(f"  Recall: {results[name]['recall']:.4f}")
            logger.info(f"  F1-Score: {results[name]['f1_score']:.4f}")
        else:
            logger.warning(f"Could not calculate utility preservation for {name}")
    
    # Summary
    logger.info("=== EVALUATION SUMMARY ===")
    logger.info(f"Raw synthetic data quality: {valid_count}/{len(X_synthetic)} valid samples ({valid_count/len(X_synthetic)*100:.1f}%)")
    logger.info(f"Completely missing rows: {all_minus_nine_rows}")
    logger.info(f"Partially missing rows: {any_minus_nine_rows - all_minus_nine_rows}")
    
    return results, baseline_results

def main():
    """
    Main function
    """
    try:
        results, baseline_results = evaluate_raw_synthetic_data()
        logger.info("=== RAW SYNTHETIC DATA EVALUATION COMPLETED ===")
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 