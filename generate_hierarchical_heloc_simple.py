import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import format_row
import logging
import sys
import os
import random
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

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hierarchical_heloc_generation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Force CUDA to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_hierarchical_model(model_path="./gpt2_finetuned_heloc_hierarchical"):
    """
    Load the hierarchical fine-tuned model and tokenizer for HELOC
    """
    try:
        logger.info(f"Loading hierarchical HELOC model from {model_path}")

        # Check if model path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} does not exist!")
            logger.info("Falling back to regular fine-tuned HELOC model...")
            model_path = "./gpt2_finetuned_heloc"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        logger.info(f"HELOC model loaded successfully on {device}")
        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error loading hierarchical HELOC model: {str(e)}")
        raise e


def generate_synthetic_data_simple(
    model, tokenizer, real_texts, num_samples=1000, device="cuda"
):
    """
    Generate synthetic HELOC data using hierarchical fine-tuned model without discriminators
    """
    logger.info(f"Generating {num_samples} synthetic HELOC samples...")

    synthetic_texts = []
    model.eval()

    for i in range(num_samples):
        try:
            # Sample a random real text as starting point
            start_text = random.choice(real_texts)

            # Generate synthetic text
            input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 20,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    num_return_sequences=1,
                )

            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            synthetic_texts.append(generated_text)

            if i % 100 == 0:
                logger.info(f"Generated {i}/{num_samples} HELOC samples")

        except Exception as e:
            logger.warning(f"Error generating HELOC sample {i}: {str(e)}")
            continue

    logger.info(f"Generated {len(synthetic_texts)} synthetic HELOC samples")
    return synthetic_texts


def parse_generated_text_to_dict(text):
    """
    Parse generated HELOC text back to dictionary format
    """
    try:
        # Split by comma and parse each field
        fields = text.split(", ")
        data_dict = {}

        for field in fields:
            if " is " in field:
                key, value = field.split(" is ", 1)
                key = key.strip()
                value = value.strip()

                # Try to convert to appropriate data type
                try:
                    # Try to convert to float first
                    if "." in value:
                        data_dict[key] = float(value)
                    else:
                        data_dict[key] = int(value)
                except ValueError:
                    # Keep as string if conversion fails
                    data_dict[key] = value

        return data_dict
    except Exception as e:
        logger.warning(f"Error parsing HELOC text: {text}, Error: {e}")
        return None


def generate_synthetic_dataset_simple(
    model, tokenizer, real_texts, device, output_file="output_hierarchical_heloc.csv"
):
    """
    Generate complete synthetic HELOC dataset and save to CSV
    """
    logger.info("Generating synthetic HELOC dataset...")

    # Generate synthetic texts
    synthetic_texts = generate_synthetic_data_simple(
        model, tokenizer, real_texts, num_samples=len(real_texts), device=device
    )

    if len(synthetic_texts) == 0:
        logger.error("No synthetic HELOC texts generated!")
        return None

    # Parse synthetic texts to dictionaries
    synthetic_data = []
    for i, text in enumerate(synthetic_texts):
        parsed = parse_generated_text_to_dict(text)
        if parsed is not None:
            synthetic_data.append(parsed)
        else:
            logger.warning(f"Failed to parse synthetic HELOC text {i}")

    logger.info(
        f"Successfully parsed {len(synthetic_data)} out of {len(synthetic_texts)} synthetic HELOC texts"
    )

    # Convert to DataFrame
    if synthetic_data:
        df_synthetic = pd.DataFrame(synthetic_data)

        # Ensure all columns from original HELOC dataset are present
        original_columns = [
            "ExternalRiskEstimate",
            "MSinceOldestTradeOpen",
            "MSinceMostRecentTradeOpen",
            "AverageMInFile",
            "NumSatisfactoryTrades",
            "NumTrades60Ever2DerogPubRec",
            "NumTrades90Ever2DerogPubRec",
            "PercentTradesNeverDelq",
            "MSinceMostRecentDelq",
            "MaxDelq2PublicRecLast12M",
            "MaxDelqEver",
            "NumTotalTrades",
            "NumTradesOpeninLast12M",
            "PercentInstallTrades",
            "MSinceMostRecentInqexcl7days",
            "NumInqLast6M",
            "NumInqLast6Mexcl7days",
            "NetFractionRevolvingBurden",
            "NetFractionInstallBurden",
            "NumRevolvingTradesWBalance",
            "NumInstallTradesWBalance",
            "NumBank2NatlTradesWHighUtilization",
            "PercentTradesWBalance",
            "RiskPerformance",
        ]

        for col in original_columns:
            if col not in df_synthetic.columns:
                df_synthetic[col] = np.nan

        # Reorder columns to match original
        df_synthetic = df_synthetic[original_columns]

        # Save to CSV
        df_synthetic.to_csv(output_file, index=False)
        logger.info(
            f"Synthetic HELOC dataset saved to {output_file} with {len(df_synthetic)} samples"
        )

        return df_synthetic
    else:
        logger.error("No valid synthetic HELOC data generated")
        return None


def clean_synthetic_data(df, output_file="output_hierarchical_heloc_clean.csv"):
    """
    Clean synthetic HELOC data by removing invalid rows and fixing data types
    """
    logger.info("Cleaning synthetic HELOC data...")

    initial_count = len(df)
    logger.info(f"Initial synthetic HELOC data count: {initial_count}")

    # Remove rows with NaN values
    df_clean = df.dropna()
    logger.info(f"After removing NaN values: {len(df_clean)} samples")

    # Convert data types
    numeric_columns = [
        "ExternalRiskEstimate",
        "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "NumTrades60Ever2DerogPubRec",
        "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "MaxDelqEver",
        "NumTotalTrades",
        "NumTradesOpeninLast12M",
        "PercentInstallTrades",
        "MSinceMostRecentInqexcl7days",
        "NumInqLast6M",
        "NumInqLast6Mexcl7days",
        "NetFractionRevolvingBurden",
        "NetFractionInstallBurden",
        "NumRevolvingTradesWBalance",
        "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
        "PercentTradesWBalance",
    ]

    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to numeric, coerce errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Handle missing values (-9, -8, -7) by replacing with NaN
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace([-9, -8, -7], np.nan)

    # Drop rows that have all -9 values (completely missing data)
    # Check if all numeric columns are -9 (which we just converted to NaN)
    all_missing_mask = df_clean[numeric_columns].isna().all(axis=1)
    df_clean = df_clean[~all_missing_mask]
    logger.info(f"After dropping completely missing rows: {len(df_clean)} samples")

    # Remove rows with too many missing values (more than 50% missing)
    missing_threshold = len(numeric_columns) * 0.5
    df_clean = df_clean.dropna(thresh=len(df_clean.columns) - missing_threshold)
    logger.info(f"After handling missing values: {len(df_clean)} samples")

    # For remaining missing values, fill with median for numeric columns
    for col in numeric_columns:
        if col in df_clean.columns and df_clean[col].isna().any():
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0  # Fallback to 0 if median is also NaN
            df_clean[col] = df_clean[col].fillna(median_val)
            logger.info(f"Filled missing values in {col} with median: {median_val}")

    # Ensure RiskPerformance is binary (Good/Bad)
    if "RiskPerformance" in df_clean.columns:
        # Convert to binary: Good=0, Bad=1
        df_clean["RiskPerformance"] = df_clean["RiskPerformance"].map({"Good": 0, "Bad": 1})
        # Handle any remaining non-binary values
        df_clean = df_clean[df_clean["RiskPerformance"].isin([0, 1])]
        logger.info(f"After RiskPerformance validation: {len(df_clean)} samples")

    # Reorder columns to match real HELOC dataset (RiskPerformance first)
    correct_column_order = [
        "RiskPerformance",
        "ExternalRiskEstimate",
        "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "NumTrades60Ever2DerogPubRec",
        "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "MaxDelqEver",
        "NumTotalTrades",
        "NumTradesOpeninLast12M",
        "PercentInstallTrades",
        "MSinceMostRecentInqexcl7days",
        "NumInqLast6M",
        "NumInqLast6Mexcl7days",
        "NetFractionRevolvingBurden",
        "NetFractionInstallBurden",
        "NumRevolvingTradesWBalance",
        "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
        "PercentTradesWBalance",
    ]
    
    # Ensure all columns exist and reorder
    for col in correct_column_order:
        if col not in df_clean.columns:
            df_clean[col] = np.nan
    
    df_clean = df_clean[correct_column_order]
    logger.info(f"After reordering columns: {len(df_clean)} samples")

    # Save cleaned data
    df_clean.to_csv(output_file, index=False)
    logger.info(
        f"Cleaned synthetic HELOC data saved to {output_file} with {len(df_clean)} samples"
    )
    logger.info(f"Data retention rate: {len(df_clean)/initial_count*100:.1f}%")

    return df_clean


def evaluate_hierarchical_synthetic_data(
    real_csv="heloc.csv", synthetic_csv="output_hierarchical_heloc_clean.csv"
):
    """
    Comprehensive evaluation of hierarchical synthetic HELOC data (no data manipulation)
    """
    logger.info("Evaluating hierarchical synthetic HELOC data...")

    try:
        # Load data
        df_real = pd.read_csv(real_csv)
        df_synthetic = pd.read_csv(synthetic_csv)

        logger.info(f"Real HELOC data: {len(df_real)} samples")
        logger.info(f"Synthetic HELOC data: {len(df_synthetic)} samples")

        # Check data quality without manipulation
        logger.info("=== DATA QUALITY ASSESSMENT ===")
        
        # Check for missing values in synthetic data
        synthetic_missing = df_synthetic.isnull().sum().sum()
        logger.info(f"Missing values in synthetic data: {synthetic_missing}")
        
        # Check for -9 values in synthetic data (missing value indicators)
        synthetic_minus_nine = (df_synthetic == -9).sum().sum()
        logger.info(f"-9 values in synthetic data: {synthetic_minus_nine}")

        # Split real data
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
            logger.info(f"Training {name} on synthetic HELOC data...")

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

                # Print detailed classification report
                print(f"\n{name} Classification Report:")
                print(classification_report(y_test, y_pred))

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = None

        # Compare with baseline (train on real data)
        logger.info("Training baseline models on real HELOC data...")
        baseline_results = {}

        for name, model in models.items():
            try:
                # Train on real training data
                model.fit(X_train, y_train)

                # Predict on real test data
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                baseline_results[name] = accuracy

                logger.info(f"{name} Baseline (Real HELOC Data) - Accuracy: {accuracy:.4f}")

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

    except Exception as e:
        logger.error(f"Error in HELOC evaluation: {str(e)}")
        raise e


def main():
    """
    Main function to run hierarchical HELOC generation and evaluation
    """
    try:
        logger.info("=== STARTING HIERARCHICAL HELOC SYNTHETIC DATA GENERATION (SIMPLE) ===")

        # Load hierarchical model
        model, tokenizer, device = load_hierarchical_model()

        # Load real HELOC data
        logger.info("Loading real HELOC data...")
        df_real = pd.read_csv("heloc.csv")
        real_texts = df_real.apply(format_row, axis=1).tolist()
        logger.info(f"Loaded {len(real_texts)} real HELOC samples")

        # Generate synthetic dataset (without discriminators for speed)
        logger.info("Generating synthetic HELOC dataset...")
        synthetic_df = generate_synthetic_dataset_simple(
            model, tokenizer, real_texts, device
        )

        if synthetic_df is not None and len(synthetic_df) > 0:
            # Clean synthetic data
            logger.info("Cleaning synthetic HELOC data...")
            clean_df = clean_synthetic_data(synthetic_df)

            if len(clean_df) > 0:
                # Evaluate synthetic data
                logger.info("Evaluating synthetic HELOC data...")
                results, baseline_results = evaluate_hierarchical_synthetic_data()

                # Print summary
                logger.info("=== HIERARCHICAL HELOC SYNTHETIC DATA EVALUATION SUMMARY ===")
                for name, metrics in results.items():
                    baseline_acc = baseline_results[name]
                    utility_preservation = (metrics["accuracy"] / baseline_acc) * 100
                    logger.info(f"{name}:")
                    logger.info(f"  Synthetic Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"  Baseline Accuracy: {baseline_acc:.4f}")
                    logger.info(f"  Utility Preservation: {utility_preservation:.2f}%")
                    logger.info(f"  Precision: {metrics['precision']:.4f}")
                    logger.info(f"  Recall: {metrics['recall']:.4f}")
                    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")

                logger.info("=== HIERARCHICAL HELOC GENERATION COMPLETED SUCCESSFULLY ===")
            else:
                logger.error("No valid synthetic HELOC data after cleaning")
        else:
            logger.error("Failed to generate synthetic HELOC data")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise e


if __name__ == "__main__":
    main() 