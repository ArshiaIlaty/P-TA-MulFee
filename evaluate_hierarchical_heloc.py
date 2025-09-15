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

def load_and_prepare_data(real_csv="heloc.csv", synthetic_csv="output_hierarchical_heloc_clean.csv"):
    """
    Load and prepare data for evaluation without any manipulation
    """
    logger.info("Loading data for evaluation...")
    
    # Load real data
    df_real = pd.read_csv(real_csv)
    logger.info(f"Real HELOC data: {len(df_real)} samples")
    
    # Load synthetic data
    df_synthetic = pd.read_csv(synthetic_csv)
    logger.info(f"Synthetic HELOC data: {len(df_synthetic)} samples")
    
    # Check data quality without manipulation
    logger.info("=== DATA QUALITY ASSESSMENT ===")
    
    # Check for missing values in real data
    real_missing = df_real.isnull().sum().sum()
    logger.info(f"Missing values in real data: {real_missing}")
    
    # Check for missing values in synthetic data
    synthetic_missing = df_synthetic.isnull().sum().sum()
    logger.info(f"Missing values in synthetic data: {synthetic_missing}")
    
    # Check for -9 values in synthetic data (missing value indicators)
    synthetic_minus_nine = (df_synthetic == -9).sum().sum()
    logger.info(f"-9 values in synthetic data: {synthetic_minus_nine}")
    
    # Check target distribution
    logger.info("Real data target distribution:")
    print(df_real["RiskPerformance"].value_counts())
    
    logger.info("Synthetic data target distribution:")
    print(df_synthetic["RiskPerformance"].value_counts())
    
    return df_real, df_synthetic

def evaluate_synthetic_data_quality(df_real, df_synthetic):
    """
    Evaluate synthetic data quality without manipulation
    """
    logger.info("=== SYNTHETIC DATA QUALITY EVALUATION ===")
    
    # Basic statistics comparison
    logger.info("Feature statistics comparison:")
    
    # Compare numeric features (excluding target)
    numeric_features = [col for col in df_real.columns if col != "RiskPerformance"]
    
    for feature in numeric_features:
        if feature in df_real.columns and feature in df_synthetic.columns:
            real_mean = df_real[feature].mean()
            real_std = df_real[feature].std()
            synthetic_mean = df_synthetic[feature].mean()
            synthetic_std = df_synthetic[feature].std()
            
            logger.info(f"{feature}:")
            logger.info(f"  Real - Mean: {real_mean:.2f}, Std: {real_std:.2f}")
            logger.info(f"  Synthetic - Mean: {synthetic_mean:.2f}, Std: {synthetic_std:.2f}")
            logger.info(f"  Difference - Mean: {abs(real_mean - synthetic_mean):.2f}, Std: {abs(real_std - synthetic_std):.2f}")

def evaluate_utility_preservation(df_real, df_synthetic):
    """
    Evaluate utility preservation by training on synthetic and testing on real
    """
    logger.info("=== UTILITY PRESERVATION EVALUATION ===")
    
    # Split real data for baseline
    X_real, y_real = df_real.drop("RiskPerformance", axis=1), df_real["RiskPerformance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    
    # Prepare synthetic data (no manipulation)
    X_synthetic, y_synthetic = (
        df_synthetic.drop("RiskPerformance", axis=1),
        df_synthetic["RiskPerformance"],
    )
    
    # Check for any remaining missing values or invalid data
    logger.info("Checking synthetic data for evaluation...")
    
    # Count rows with any missing values or -9 values
    missing_mask = X_synthetic.isnull().any(axis=1)
    minus_nine_mask = (X_synthetic == -9).any(axis=1)
    invalid_mask = missing_mask | minus_nine_mask
    
    invalid_count = invalid_mask.sum()
    valid_count = (~invalid_mask).sum()
    
    logger.info(f"Synthetic data quality for evaluation:")
    logger.info(f"  Valid rows (no missing/-9): {valid_count}")
    logger.info(f"  Invalid rows (with missing/-9): {invalid_count}")
    logger.info(f"  Valid percentage: {valid_count/len(X_synthetic)*100:.1f}%")
    
    if valid_count == 0:
        logger.error("No valid synthetic data for evaluation!")
        return None, None
    
    # Use only valid synthetic data
    X_synthetic_valid = X_synthetic[~invalid_mask]
    y_synthetic_valid = y_synthetic[~invalid_mask]
    
    logger.info(f"Using {len(X_synthetic_valid)} valid synthetic samples for evaluation")
    
    # Train models on synthetic data and test on real data
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name} on synthetic data...")
        
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
    
    # Compare with baseline (train on real data)
    logger.info("Training baseline models on real data...")
    baseline_results = {}
    
    for name, model in models.items():
        try:
            # Train on real training data
            model.fit(X_train, y_train)
            
            # Predict on real test data
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            baseline_results[name] = accuracy
            
            logger.info(f"{name} Baseline (Real Data) - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error training baseline {name}: {str(e)}")
            baseline_results[name] = None
    
    # Calculate utility preservation
    logger.info("Calculating utility preservation...")
    for name in results.keys():
        if results[name] is not None and baseline_results[name] is not None:
            synthetic_acc = results[name]["accuracy"]
            baseline_acc = baseline_results[name]
            utility_preservation = (synthetic_acc / baseline_acc) * 100
            
            logger.info(f"{name} Utility Preservation: {utility_preservation:.2f}%")
        else:
            logger.warning(f"Could not calculate utility preservation for {name}")
    
    return results, baseline_results

def main():
    """
    Main evaluation function
    """
    logger.info("=== HIERARCHICAL HELOC SYNTHETIC DATA EVALUATION ===")
    
    try:
        # Load and prepare data
        df_real, df_synthetic = load_and_prepare_data()
        
        # Evaluate synthetic data quality
        evaluate_synthetic_data_quality(df_real, df_synthetic)
        
        # Evaluate utility preservation
        results, baseline_results = evaluate_utility_preservation(df_real, df_synthetic)
        
        if results is not None:
            # Print summary
            logger.info("=== EVALUATION SUMMARY ===")
            for name, metrics in results.items():
                if metrics is not None and baseline_results[name] is not None:
                    baseline_acc = baseline_results[name]
                    utility_preservation = (metrics["accuracy"] / baseline_acc) * 100
                    logger.info(f"{name}:")
                    logger.info(f"  Synthetic Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"  Baseline Accuracy: {baseline_acc:.4f}")
                    logger.info(f"  Utility Preservation: {utility_preservation:.2f}%")
                    logger.info(f"  Precision: {metrics['precision']:.4f}")
                    logger.info(f"  Recall: {metrics['recall']:.4f}")
                    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        logger.info("=== EVALUATION COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 