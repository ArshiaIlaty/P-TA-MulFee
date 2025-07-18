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
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hierarchical_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Force CUDA to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_model_and_tokenizer(model_path="./gpt2_finetuned_diabetes"):
    """
    Load the fine-tuned model and tokenizer
    """
    try:
        logger.info(f"Loading model and tokenizer from {model_path}")

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

        logger.info(f"Model and tokenizer loaded successfully on {device}")
        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise e


def generate_synthetic_samples_for_training(
    model, tokenizer, real_texts, num_samples=1000, device="cuda"
):
    """
    Generate synthetic samples for training discriminators
    """
    logger.info(
        f"Generating {num_samples} synthetic samples for discriminator training..."
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
            synthetic_texts.append(generated_text)

            if i % 100 == 0:
                logger.info(f"Generated {i}/{num_samples} synthetic samples")

        except Exception as e:
            logger.warning(f"Error generating sample {i}: {str(e)}")
            continue

    logger.info(f"Successfully generated {len(synthetic_texts)} synthetic samples")
    return synthetic_texts


def prepare_training_data(real_texts, synthetic_texts):
    """
    Prepare training data for discriminators
    """
    logger.info("Preparing training data for discriminators...")

    # Create labels: 1 for real, 0 for synthetic
    real_labels = [1] * len(real_texts)
    synthetic_labels = [0] * len(synthetic_texts)

    # Combine data
    all_texts = real_texts + synthetic_texts
    all_labels = real_labels + synthetic_labels

    # Shuffle data
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)

    # Split into train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    logger.info(f"Training data: {len(train_texts)} samples")
    logger.info(f"Validation data: {len(val_texts)} samples")
    logger.info(f"Real samples in training: {sum(train_labels)}")
    logger.info(
        f"Synthetic samples in training: {len(train_labels) - sum(train_labels)}"
    )

    return train_texts, val_texts, train_labels, val_labels


def train_hierarchical_discriminators(
    real_csv="diabetes.csv",
    model_path="./gpt2_finetuned_diabetes",
    save_path="./hierarchical_discriminators",
    num_synthetic_samples=1000,
):
    """
    Train hierarchical discriminators
    """
    try:
        logger.info("=== STARTING HIERARCHICAL DISCRIMINATOR TRAINING ===")

        # Load real data
        logger.info(f"Loading real data from {real_csv}")
        df_real = pd.read_csv(real_csv)
        real_texts = df_real.apply(format_row, axis=1).tolist()
        logger.info(f"Loaded {len(real_texts)} real samples")

        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer(model_path)

        # Generate synthetic samples
        synthetic_texts = generate_synthetic_samples_for_training(
            model, tokenizer, real_texts, num_synthetic_samples, device
        )

        # Prepare training data
        train_texts, val_texts, train_labels, val_labels = prepare_training_data(
            real_texts, synthetic_texts
        )

        # Initialize hierarchical discriminators
        logger.info("Initializing hierarchical discriminators...")
        hierarchical_discriminators = HierarchicalDiscriminatorSystem(device=device)

        # Train discriminators
        logger.info("Training hierarchical discriminators...")
        training_results = hierarchical_discriminators.train_discriminators(
            real_texts=train_texts,
            synthetic_texts=[
                text for text, label in zip(train_texts, train_labels) if label == 0
            ],
            epochs=3,
        )

        # Save discriminators
        logger.info(f"Saving discriminators to {save_path}")
        hierarchical_discriminators.save_discriminators(save_path)

        # Test discriminators on validation set
        logger.info("Testing discriminators on validation set...")
        correct_predictions = 0
        total_predictions = 0

        for text, label in zip(val_texts, val_labels):
            try:
                feedback = hierarchical_discriminators.get_multi_level_feedback(text)

                # Use row-level prediction as main prediction
                prediction = 1 if feedback["row"] > 0.5 else 0

                if prediction == label:
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                logger.warning(f"Error evaluating sample: {str(e)}")
                continue

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        logger.info(
            f"Validation accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})"
        )

        # Test on a few samples
        logger.info("Testing discriminators on sample texts...")
        test_samples = random.sample(real_texts, 3) + random.sample(synthetic_texts, 3)

        for i, text in enumerate(test_samples):
            try:
                feedback = hierarchical_discriminators.get_multi_level_feedback(text)
                logger.info(f"Sample {i+1} feedback:")
                logger.info(f"  Token level: {feedback['token']:.3f}")
                logger.info(f"  Sentence level: {feedback['sentence']:.3f}")
                logger.info(f"  Row level: {feedback['row']:.3f}")
                logger.info(f"  Features: {feedback['features']}")
                logger.info(f"  Text preview: {text[:100]}...")
                logger.info("---")

            except Exception as e:
                logger.warning(f"Error testing sample {i+1}: {str(e)}")

        logger.info("=== HIERARCHICAL DISCRIMINATOR TRAINING COMPLETED ===")
        logger.info(f"Discriminators saved to: {save_path}")

        return hierarchical_discriminators, accuracy

    except Exception as e:
        logger.error(f"Error in training hierarchical discriminators: {str(e)}")
        raise e


def main():
    """
    Main function to train hierarchical discriminators
    """
    try:
        logger.info("Starting hierarchical discriminator training script...")

        # Train discriminators
        discriminators, accuracy = train_hierarchical_discriminators(
            real_csv="diabetes.csv",
            model_path="./gpt2_finetuned_diabetes",
            save_path="./hierarchical_discriminators",
            num_synthetic_samples=1000,
        )

        logger.info(
            f"Training completed successfully with validation accuracy: {accuracy:.4f}"
        )

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
