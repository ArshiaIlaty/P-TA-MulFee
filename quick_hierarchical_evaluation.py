import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import format_row
import logging
import sys
import os
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("quick_hierarchical_evaluation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_hierarchical_model(model_path="./gpt2_finetuned_diabetes_hierarchical"):
    """
    Load the hierarchical fine-tuned model and tokenizer
    """
    try:
        logger.info(f"Loading hierarchical model from {model_path}")

        # Check if model path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path {model_path} does not exist!")
            logger.info("Falling back to regular fine-tuned model...")
            model_path = "./gpt2_finetuned_diabetes"

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

        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error loading hierarchical model: {str(e)}")
        raise e


def generate_sample_with_hierarchical(model, tokenizer, real_text, device="cuda"):
    """
    Generate a single sample using the hierarchical model
    """
    try:
        # Generate synthetic text
        input_ids = tokenizer.encode(real_text, return_tensors="pt").to(device)

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

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        logger.warning(f"Error generating sample: {str(e)}")
        return None


def test_hierarchical_model_quality(
    model, tokenizer, real_texts, device, num_samples=100
):
    """
    Test the quality of the hierarchical model by generating samples and comparing with existing synthetic data
    """
    logger.info(f"Testing hierarchical model quality with {num_samples} samples...")

    # Generate new samples with hierarchical model
    hierarchical_samples = []
    for i in range(num_samples):
        if i % 10 == 0:
            logger.info(f"Generating sample {i}/{num_samples}")

        real_text = real_texts[i % len(real_texts)]  # Cycle through real texts
        generated_text = generate_sample_with_hierarchical(
            model, tokenizer, real_text, device
        )

        if generated_text:
            hierarchical_samples.append(generated_text)

    logger.info(f"Generated {len(hierarchical_samples)} hierarchical samples")

    # Compare with existing synthetic data
    try:
        existing_synthetic = pd.read_csv("output_diabetes_clean.csv")
        logger.info(f"Existing synthetic data: {len(existing_synthetic)} samples")

        # Show a few examples
        logger.info("=== SAMPLE COMPARISON ===")
        logger.info("Real text example:")
        logger.info(f"  {real_texts[0]}")
        logger.info("Hierarchical generated example:")
        logger.info(f"  {hierarchical_samples[0] if hierarchical_samples else 'None'}")
        logger.info("Existing synthetic example:")
        existing_text = existing_synthetic.iloc[0].apply(
            lambda x: f"{existing_synthetic.columns[existing_synthetic.columns.get_loc(x.name)]} is {x}"
        )
        logger.info(f"  {', '.join(existing_text)}")

    except Exception as e:
        logger.warning(f"Could not load existing synthetic data: {str(e)}")

    return hierarchical_samples


def evaluate_existing_synthetic_data():
    """
    Evaluate the existing synthetic data (output_diabetes_clean.csv)
    """
    logger.info("Evaluating existing synthetic data...")

    try:
        # Load data
        df_real = pd.read_csv("diabetes.csv")
        df_synthetic = pd.read_csv("output_diabetes_clean.csv")

        logger.info(f"Real data: {len(df_real)} samples")
        logger.info(f"Synthetic data: {len(df_synthetic)} samples")

        # Split real data
        X_real, y_real = df_real.drop("diabetes", axis=1), df_real["diabetes"]
        X_train, X_test, y_train, y_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42
        )

        # Prepare synthetic data
        X_synthetic, y_synthetic = (
            df_synthetic.drop("diabetes", axis=1),
            df_synthetic["diabetes"],
        )

        # Handle categorical variables
        categorical_columns = ["gender", "smoking_history"]

        for col in categorical_columns:
            if col in X_train.columns:
                # Combine all unique values
                all_values = set(X_train[col].unique()) | set(X_synthetic[col].unique())

                # Create mapping
                value_mapping = {val: idx for idx, val in enumerate(all_values)}

                # Apply mapping
                X_train[col] = X_train[col].map(value_mapping)
                X_test[col] = X_test[col].map(value_mapping)
                X_synthetic[col] = X_synthetic[col].map(value_mapping)

        # Train models on synthetic data and test on real data
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        }

        results = {}

        for name, model in models.items():
            logger.info(f"Training {name} on synthetic data...")

            # Train on synthetic data
            model.fit(X_synthetic, y_synthetic)

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

        # Compare with baseline (train on real data)
        logger.info("Training baseline models on real data...")
        baseline_results = {}

        for name, model in models.items():
            # Train on real training data
            model.fit(X_train, y_train)

            # Predict on real test data
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            baseline_results[name] = accuracy

            logger.info(f"{name} Baseline (Real Data) - Accuracy: {accuracy:.4f}")

        # Calculate utility preservation
        logger.info("Calculating utility preservation...")
        for name in results.keys():
            synthetic_acc = results[name]["accuracy"]
            baseline_acc = baseline_results[name]
            utility_preservation = (synthetic_acc / baseline_acc) * 100

            logger.info(f"{name} Utility Preservation: {utility_preservation:.2f}%")

        return results, baseline_results

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise e


def main():
    """
    Main function to quickly test hierarchical model
    """
    try:
        logger.info("=== QUICK HIERARCHICAL MODEL EVALUATION ===")

        # Load hierarchical model
        model, tokenizer, device = load_hierarchical_model()

        # Load real data
        logger.info("Loading real data...")
        df_real = pd.read_csv("diabetes.csv")
        real_texts = df_real.apply(format_row, axis=1).tolist()
        logger.info(f"Loaded {len(real_texts)} real samples")

        # Test hierarchical model quality (generate a few samples)
        logger.info("Testing hierarchical model quality...")
        hierarchical_samples = test_hierarchical_model_quality(
            model, tokenizer, real_texts, device, num_samples=10
        )

        # Evaluate existing synthetic data
        logger.info("Evaluating existing synthetic data...")
        results, baseline_results = evaluate_existing_synthetic_data()

        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")
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

        logger.info("=== QUICK EVALUATION COMPLETED ===")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
