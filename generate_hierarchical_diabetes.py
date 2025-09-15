import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
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
        logging.FileHandler("hierarchical_generation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Force CUDA to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def load_hierarchical_discriminators(
    discriminator_path="./hierarchical_discriminators",
):
    """
    Load hierarchical discriminators with fallback
    """
    try:
        logger.info(f"Loading hierarchical discriminators from {discriminator_path}")

        if not os.path.exists(discriminator_path):
            logger.warning(f"Discriminator path {discriminator_path} does not exist!")
            logger.info("Creating new hierarchical discriminators...")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            discriminators = HierarchicalDiscriminatorSystem(device=device, dataset_type="diabetes")
            logger.info("New hierarchical discriminators created (untrained)")
            return discriminators, False  # False indicates not trained

        # Load existing discriminators
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        discriminators = HierarchicalDiscriminatorSystem(device=device, dataset_type="diabetes")
        discriminators.load_discriminators(discriminator_path)
        logger.info("Hierarchical discriminators loaded successfully")
        return discriminators, True  # True indicates trained

    except Exception as e:
        logger.error(f"Error loading hierarchical discriminators: {str(e)}")
        logger.info("Creating new hierarchical discriminators as fallback...")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        discriminators = HierarchicalDiscriminatorSystem(device=device, dataset_type="diabetes")
        return discriminators, False


def generate_synthetic_data_hierarchical(
    model,
    tokenizer,
    real_texts,
    hierarchical_discriminators,
    num_samples=1000,
    device="cuda",
    use_discriminators=True,
):
    """
    Generate synthetic data using hierarchical fine-tuned model with discriminator feedback
    """
    logger.info(
        f"Generating {num_samples} synthetic samples with hierarchical feedback..."
    )

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

            # Get hierarchical feedback if discriminators are available and trained
            if use_discriminators:
                try:
                    feedback = hierarchical_discriminators.get_multi_level_feedback(
                        generated_text
                    )

                    # Calculate quality score
                    quality_score = (
                        feedback["token"] * 0.2
                        + feedback["sentence"] * 0.3
                        + feedback["row"] * 0.3
                        + np.mean(list(feedback["features"].values())) * 0.2
                    )

                    # Only keep high-quality synthetic data
                    if quality_score > 0.5:  # Lowered threshold for more samples
                        synthetic_texts.append(generated_text)
                        logger.debug(
                            f"Sample {i} accepted with quality score: {quality_score:.3f}"
                        )
                    else:
                        logger.debug(
                            f"Sample {i} rejected with quality score: {quality_score:.3f}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Error getting discriminator feedback for sample {i}: {str(e)}"
                    )
                    # Accept the sample anyway if discriminator fails
                    synthetic_texts.append(generated_text)
            else:
                # If no discriminators, accept all samples
                synthetic_texts.append(generated_text)

            if i % 100 == 0:
                logger.info(
                    f"Generated {i}/{num_samples} samples, accepted: {len(synthetic_texts)}"
                )

        except Exception as e:
            logger.warning(f"Error generating sample {i}: {str(e)}")
            continue

    logger.info(
        f"Generated {len(synthetic_texts)} synthetic samples out of {num_samples} attempts"
    )
    return synthetic_texts


def parse_generated_text_to_dict(text):
    """
    Parse generated text back to dictionary format
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
        logger.warning(f"Error parsing text: {text}, Error: {e}")
        return None


def generate_synthetic_dataset(
    model,
    tokenizer,
    real_texts,
    hierarchical_discriminators,
    device,
    output_file="output_hierarchical.csv",
    use_discriminators=True,
):
    """
    Generate complete synthetic dataset and save to CSV
    """
    logger.info("Generating synthetic dataset...")

    # Generate synthetic texts
    synthetic_texts = generate_synthetic_data_hierarchical(
        model,
        tokenizer,
        real_texts,
        hierarchical_discriminators,
        num_samples=len(real_texts),
        device=device,
        use_discriminators=use_discriminators,
    )

    if len(synthetic_texts) == 0:
        logger.error("No synthetic texts generated!")
        return None

    # Parse synthetic texts to dictionaries
    synthetic_data = []
    for i, text in enumerate(synthetic_texts):
        parsed = parse_generated_text_to_dict(text)
        if parsed is not None:
            synthetic_data.append(parsed)
        else:
            logger.warning(f"Failed to parse synthetic text {i}")

    logger.info(
        f"Successfully parsed {len(synthetic_data)} out of {len(synthetic_texts)} synthetic texts"
    )

    # Convert to DataFrame
    if synthetic_data:
        df_synthetic = pd.DataFrame(synthetic_data)

        # Ensure all columns from original dataset are present
        original_columns = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
            "diabetes",
        ]

        for col in original_columns:
            if col not in df_synthetic.columns:
                df_synthetic[col] = np.nan

        # Reorder columns to match original
        df_synthetic = df_synthetic[original_columns]

        # Save to CSV
        df_synthetic.to_csv(output_file, index=False)
        logger.info(
            f"Synthetic dataset saved to {output_file} with {len(df_synthetic)} samples"
        )

        return df_synthetic
    else:
        logger.error("No valid synthetic data generated")
        return None


def clean_synthetic_data(df, output_file="output_hierarchical_clean.csv"):
    """
    Clean synthetic data by removing invalid rows and fixing data types
    """
    logger.info("Cleaning synthetic data...")

    initial_count = len(df)
    logger.info(f"Initial synthetic data count: {initial_count}")

    # Remove rows with NaN values
    df_clean = df.dropna()
    logger.info(f"After removing NaN values: {len(df_clean)} samples")

    # Convert data types
    numeric_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "diabetes",
    ]

    for col in numeric_columns:
        if col in df_clean.columns:
            # Convert to numeric, coerce errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Remove rows with invalid numeric values
    df_clean = df_clean.dropna()
    logger.info(f"After numeric validation: {len(df_clean)} samples")

    # Ensure categorical columns have valid values
    if "gender" in df_clean.columns:
        valid_genders = ["Male", "Female"]
        df_clean = df_clean[df_clean["gender"].isin(valid_genders)]
        logger.info(f"After gender validation: {len(df_clean)} samples")

    if "smoking_history" in df_clean.columns:
        valid_smoking = ["never", "current", "former", "ever", "not current"]
        df_clean = df_clean[df_clean["smoking_history"].isin(valid_smoking)]
        logger.info(f"After smoking history validation: {len(df_clean)} samples")

    # Save cleaned data
    df_clean.to_csv(output_file, index=False)
    logger.info(
        f"Cleaned synthetic data saved to {output_file} with {len(df_clean)} samples"
    )
    logger.info(f"Data retention rate: {len(df_clean)/initial_count*100:.1f}%")

    return df_clean


def evaluate_hierarchical_synthetic_data(
    real_csv="diabetes.csv", synthetic_csv="output_hierarchical_clean.csv"
):
    """
    Comprehensive evaluation of hierarchical synthetic data
    """
    logger.info("Evaluating hierarchical synthetic data...")

    try:
        # Load data
        df_real = pd.read_csv(real_csv)
        df_synthetic = pd.read_csv(synthetic_csv)

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

            # Print detailed classification report
            print(f"\n{name} Classification Report:")
            print(classification_report(y_test, y_pred))

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
    Main function to run hierarchical generation and evaluation
    """
    try:
        logger.info("=== STARTING HIERARCHICAL SYNTHETIC DATA GENERATION ===")

        # Load hierarchical model
        model, tokenizer, device = load_hierarchical_model()

        # Load real data
        logger.info("Loading real data...")
        df_real = pd.read_csv("diabetes.csv")
        real_texts = df_real.apply(format_row, axis=1).tolist()
        logger.info(f"Loaded {len(real_texts)} real samples")

        # Load hierarchical discriminators
        logger.info("Loading hierarchical discriminators...")
        hierarchical_discriminators, discriminators_trained = (
            load_hierarchical_discriminators()
        )

        if discriminators_trained:
            logger.info("Using trained hierarchical discriminators")
            use_discriminators = True
        else:
            logger.warning(
                "Using untrained hierarchical discriminators - quality filtering disabled"
            )
            use_discriminators = False

        # Generate synthetic dataset
        logger.info("Generating synthetic dataset...")
        synthetic_df = generate_synthetic_dataset(
            model,
            tokenizer,
            real_texts,
            hierarchical_discriminators,
            device,
            use_discriminators=use_discriminators,
        )

        if synthetic_df is not None and len(synthetic_df) > 0:
            # Clean synthetic data
            logger.info("Cleaning synthetic data...")
            clean_df = clean_synthetic_data(synthetic_df)

            if len(clean_df) > 0:
                # Evaluate synthetic data
                logger.info("Evaluating synthetic data...")
                results, baseline_results = evaluate_hierarchical_synthetic_data()

                # Print summary
                logger.info("=== HIERARCHICAL SYNTHETIC DATA EVALUATION SUMMARY ===")
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

                logger.info("=== HIERARCHICAL GENERATION COMPLETED SUCCESSFULLY ===")
            else:
                logger.error("No valid synthetic data after cleaning")
        else:
            logger.error("Failed to generate synthetic data")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
