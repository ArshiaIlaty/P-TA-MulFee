from great_heloc import *
from classifier_heloc import *
from gan_heloc import *
from hierarchical_discriminators import *
import torch
import random
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AdamW,
)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import format_row
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hierarchical_heloc_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def generate_synthetic_data_with_hierarchical_feedback(
    model,
    tokenizer,
    real_texts,
    hierarchical_discriminators,
    num_samples=1000,
    device="cuda",
):
    """
    Generate synthetic data using GREAT model with hierarchical discriminator feedback
    """
    logger.info(
        f"Generating {num_samples} synthetic samples with hierarchical feedback..."
    )

    synthetic_texts = []
    model.eval()

    for i in range(num_samples):
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
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Get hierarchical feedback
        feedback = hierarchical_discriminators.get_multi_level_feedback(generated_text)

        # Only keep high-quality synthetic data
        quality_score = (
            feedback["token"] * 0.2
            + feedback["sentence"] * 0.3
            + feedback["row"] * 0.3
            + np.mean(list(feedback["features"].values())) * 0.2
        )

        if quality_score > 0.6:  # Quality threshold
            synthetic_texts.append(generated_text)

        if i % 100 == 0:
            logger.info(
                f"Generated {i}/{num_samples} samples, quality score: {quality_score:.3f}"
            )

    logger.info(f"Generated {len(synthetic_texts)} high-quality synthetic samples")
    return synthetic_texts


def train_with_hierarchical_discriminators(
    csv_path="heloc.csv",
    model_name="gpt2",
    save_path="./gpt2_finetuned_heloc_hierarchical",
    classifier_save_path="./classifier_heloc_hierarchical.pth",
    N=2,
    total_epoch_num=2,
):
    """
    Enhanced training pipeline with hierarchical discriminators for HELOC dataset
    """
    device = torch.device(
        "cuda:1"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else "cuda:0"
    )
    logger.info(f"Using device: {device}")

    # Load data
    df = pd.read_csv(csv_path)
    real_texts = df.apply(format_row, axis=1).tolist()

    # Initialize hierarchical discriminator system
    logger.info("Initializing hierarchical discriminator system...")
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(device=device)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    for epoch in range(total_epoch_num):
        logger.info(f"----- EPOCH {epoch + 1}/{total_epoch_num} -----")

        # Step 1: GREAT Model Training
        logger.info("Step 1: GREAT Model Training...")
        if epoch == 0:
            train_great(csv_path, model_name, save_path)
        else:
            train_great(csv_path, save_path, save_path)

        # Load the trained GREAT model
        great_model = AutoModelForCausalLM.from_pretrained(save_path).to(device)

        # Step 2: Generate initial synthetic data
        logger.info("Step 2: Generating initial synthetic data...")
        initial_synthetic_texts = generate_synthetic_data_with_hierarchical_feedback(
            great_model,
            tokenizer,
            real_texts,
            hierarchical_discriminators,
            num_samples=len(real_texts) // 2,
        )

        # Step 3: Train hierarchical discriminators
        logger.info("Step 3: Training hierarchical discriminators...")
        hierarchical_discriminators.train_discriminators(
            real_texts, initial_synthetic_texts
        )

        # Step 3.5: Hierarchical GAN adversarial training
        logger.info(
            "Step 3.5: Hierarchical GAN adversarial training with multi-level feedback..."
        )
        train_gpt2_with_hierarchical_gan(
            df,
            great_model,
            tokenizer,
            hierarchical_discriminators,
            device,
            num_epochs=1,
            N=N,
        )

        # Step 4: Enhanced GREAT training with hierarchical feedback
        logger.info("Step 4: Enhanced GREAT training with hierarchical feedback...")
        integrate_with_great_training(
            great_model,
            tokenizer,
            hierarchical_discriminators,
            real_texts,
            initial_synthetic_texts,
            device,
        )

        # Save enhanced model
        great_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Step 5: Generate high-quality synthetic data
        logger.info("Step 5: Generating high-quality synthetic data...")
        final_synthetic_texts = generate_synthetic_data_with_hierarchical_feedback(
            great_model,
            tokenizer,
            real_texts,
            hierarchical_discriminators,
            num_samples=len(real_texts),
        )

        # Step 6: Train enhanced classifier with hierarchical feedback
        logger.info("Step 6: Training enhanced classifier...")
        enhanced_classifier = train_enhanced_classifier(
            real_texts,
            final_synthetic_texts,
            tokenizer,
            device,
            hierarchical_discriminators,
            classifier_save_path,
        )

        # Step 7: Save hierarchical discriminators
        logger.info("Step 7: Saving hierarchical discriminators...")
        hierarchical_discriminators.save_discriminators(f"{save_path}_discriminators")

        # Step 8: Generate final synthetic dataset
        logger.info("Step 8: Generating final synthetic dataset...")
        generate_final_synthetic_dataset(
            great_model,
            tokenizer,
            real_texts,
            hierarchical_discriminators,
            save_path,
            device,
        )

        logger.info(f"Epoch {epoch + 1} completed successfully!")


def train_enhanced_classifier(
    real_texts,
    synthetic_texts,
    tokenizer,
    device,
    hierarchical_discriminators,
    save_path,
):
    """
    Train enhanced classifier with hierarchical discriminator feedback
    """
    logger.info("Training enhanced classifier with hierarchical feedback...")

    # Prepare data
    real_labels = [1] * len(real_texts)
    synthetic_labels = [0] * len(synthetic_texts)

    all_texts = real_texts + synthetic_texts
    all_labels = real_labels + synthetic_labels

    # Create enhanced dataset with hierarchical feedback
    enhanced_dataset = HierarchicalDataset(
        all_texts, all_labels, tokenizer, hierarchical_discriminators
    )

    dataloader = DataLoader(enhanced_dataset, batch_size=4, shuffle=True)

    # Initialize classifier
    classifier = TextClassifier("gpt2").to(device)
    optimizer = AdamW(classifier.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    classifier.train()
    for epoch in range(3):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            hierarchical_weights = batch["hierarchical_weights"].to(device)

            optimizer.zero_grad()

            outputs = classifier(input_ids, attention_mask)
            base_loss = loss_fn(outputs, labels)

            # Apply hierarchical weights to loss
            weighted_loss = base_loss * hierarchical_weights.mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()

            if batch_idx % 50 == 0:
                logger.info(
                    f"Classifier batch {batch_idx}, loss: {weighted_loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Classifier epoch {epoch + 1}, avg loss: {avg_loss:.4f}")

    # Save classifier
    torch.save(classifier.state_dict(), save_path)
    logger.info(f"Enhanced classifier saved to {save_path}")

    return classifier


class HierarchicalDataset(Dataset):
    """
    Enhanced dataset that includes hierarchical discriminator feedback
    """

    def __init__(
        self, texts, labels, tokenizer, hierarchical_discriminators, max_length=128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.hierarchical_discriminators = hierarchical_discriminators
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Get hierarchical feedback
        feedback = self.hierarchical_discriminators.get_multi_level_feedback(text)

        # Calculate hierarchical weight based on feedback quality
        hierarchical_weight = (
            feedback["token"] * 0.2
            + feedback["sentence"] * 0.3
            + feedback["row"] * 0.3
            + np.mean(list(feedback["features"].values())) * 0.2
        )

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label),
            "hierarchical_weights": torch.tensor(hierarchical_weight),
        }


def generate_final_synthetic_dataset(
    great_model, tokenizer, real_texts, hierarchical_discriminators, save_path, device
):
    """
    Generate final high-quality synthetic dataset
    """
    logger.info("Generating final synthetic dataset...")

    synthetic_data = []
    great_model.eval()

    for i, real_text in enumerate(real_texts):
        # Generate synthetic version
        input_ids = tokenizer.encode(real_text, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = great_model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 20,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Get hierarchical feedback
        feedback = hierarchical_discriminators.get_multi_level_feedback(generated_text)

        # Only keep high-quality samples
        quality_score = (
            feedback["token"] * 0.2
            + feedback["sentence"] * 0.3
            + feedback["row"] * 0.3
            + np.mean(list(feedback["features"].values())) * 0.2
        )

        if quality_score > 0.7:  # Higher threshold for final dataset
            # Parse generated text back to tabular format
            try:
                row_data = parse_generated_text_to_dict(generated_text)
                if row_data:
                    synthetic_data.append(row_data)
            except Exception as e:
                logger.warning(f"Failed to parse generated text: {e}")

        if i % 1000 == 0:
            logger.info(
                f"Processed {i}/{len(real_texts)} samples, quality score: {quality_score:.3f}"
            )

    # Convert to DataFrame and save
    if synthetic_data:
        synthetic_df = pd.DataFrame(synthetic_data)
        output_path = f"{save_path}/output_hierarchical_heloc.csv"
        synthetic_df.to_csv(output_path, index=False)
        logger.info(
            f"Final synthetic dataset saved to {output_path} with {len(synthetic_data)} samples"
        )
    else:
        logger.warning("No high-quality synthetic data generated!")


def parse_generated_text_to_dict(text):
    """
    Parse generated text back to dictionary format
    """
    try:
        pairs = text.split(", ")
        data = {}
        for pair in pairs:
            if " is " in pair:
                key, value = pair.split(" is ", 1)
                # Clean up the key and value
                key = key.strip()
                value = value.strip()

                # Try to convert to appropriate type
                try:
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    pass  # Keep as string if conversion fails

                data[key] = value

        return data
    except Exception as e:
        logger.error(f"Error parsing text: {e}")
        return None


def evaluate_hierarchical_system(
    real_csv="heloc.csv", synthetic_csv="output_hierarchical_heloc.csv"
):
    """
    Evaluate the hierarchical system performance for HELOC dataset
    """
    logger.info("Evaluating hierarchical system performance for HELOC...")

    try:
        real_df = pd.read_csv(real_csv)
        synthetic_df = pd.read_csv(synthetic_csv)

        # Basic statistics
        logger.info(
            f"Real dataset: {len(real_df)} samples, {len(real_df.columns)} features"
        )
        logger.info(
            f"Synthetic dataset: {len(synthetic_df)} samples, {len(synthetic_df.columns)} features"
        )

        # Feature-wise comparison
        for col in real_df.columns:
            if col in synthetic_df.columns:
                real_mean = (
                    real_df[col].mean()
                    if real_df[col].dtype in ["int64", "float64"]
                    else None
                )
                synth_mean = (
                    synthetic_df[col].mean()
                    if synthetic_df[col].dtype in ["int64", "float64"]
                    else None
                )

                if real_mean is not None and synth_mean is not None:
                    logger.info(
                        f"{col}: Real mean={real_mean:.3f}, Synthetic mean={synth_mean:.3f}"
                    )

        # Run evaluation script
        from evaluate_heloc import train_and_evaluate

        train_and_evaluate(synthetic_csv, real_csv, "RiskPerformance")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    try:
        logger.info("Starting hierarchical P-TA training pipeline for HELOC dataset...")

        # Run enhanced training
        train_with_hierarchical_discriminators()

        # Evaluate results
        evaluate_hierarchical_system()

        logger.info("Hierarchical P-TA pipeline for HELOC completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e 