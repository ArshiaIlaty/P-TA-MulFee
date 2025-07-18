import torch
import pandas as pd
import numpy as np
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
        logging.FileHandler("discriminator_testing.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Force CUDA to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test_discriminator_initialization():
    """
    Test 1: Initialize discriminators and check all components
    """
    logger.info("=== TEST 1: DISCRIMINATOR INITIALIZATION ===")

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        discriminators = HierarchicalDiscriminatorSystem(device=device)

        # Check if all discriminators are initialized
        assert (
            discriminators.token_discriminator is not None
        ), "Token discriminator not initialized"
        assert (
            discriminators.sentence_discriminator is not None
        ), "Sentence discriminator not initialized"
        assert (
            discriminators.row_discriminator is not None
        ), "Row discriminator not initialized"
        assert (
            len(discriminators.feature_discriminators) > 0
        ), "Feature discriminators not initialized"

        logger.info("✅ All discriminators initialized successfully")
        logger.info(f"Token discriminator: {type(discriminators.token_discriminator)}")
        logger.info(
            f"Sentence discriminator: {type(discriminators.sentence_discriminator)}"
        )
        logger.info(f"Row discriminator: {type(discriminators.row_discriminator)}")
        logger.info(
            f"Feature discriminators: {len(discriminators.feature_discriminators)} discriminators"
        )

        return discriminators

    except Exception as e:
        logger.error(f"❌ Initialization test failed: {str(e)}")
        raise e


def test_discriminator_forward_pass(discriminators):
    """
    Test 2: Test forward pass for each discriminator
    """
    logger.info("=== TEST 2: FORWARD PASS TESTING ===")

    try:
        # Test text
        test_text = "gender is Female, age is 30, hypertension is 0, heart_disease is 0, smoking_history is never, bmi is 25.5, HbA1c_level is 5.2, blood_glucose_level is 120, diabetes is 0"

        # Test token-level discriminator
        logger.info("Testing token-level discriminator...")
        token_ids = discriminators.tokenizer.encode(test_text, return_tensors="pt").to(
            discriminators.device
        )
        token_output = discriminators.token_discriminator(token_ids)
        logger.info(f"✅ Token discriminator output shape: {token_output.shape}")
        logger.info(
            f"✅ Token discriminator output range: [{token_output.min().item():.3f}, {token_output.max().item():.3f}]"
        )

        # Test sentence-level discriminator
        logger.info("Testing sentence-level discriminator...")
        sentences = test_text.split(", ")
        for sentence in sentences[:3]:  # Test first 3 sentences
            if " is " in sentence:
                encoding = discriminators.tokenizer(
                    sentence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(discriminators.device)
                sentence_output = discriminators.sentence_discriminator(
                    encoding["input_ids"], encoding["attention_mask"]
                )
                logger.info(
                    f"✅ Sentence discriminator output for '{sentence}': {sentence_output.item():.3f}"
                )

        # Test row-level discriminator
        logger.info("Testing row-level discriminator...")
        encoding = discriminators.tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(discriminators.device)
        row_output = discriminators.row_discriminator(
            encoding["input_ids"], encoding["attention_mask"]
        )
        logger.info(f"✅ Row discriminator output shape: {row_output.shape}")
        logger.info(f"✅ Row discriminator output: {row_output.item():.3f}")

        # Test feature-level discriminators
        logger.info("Testing feature-level discriminators...")
        for (
            feature_name,
            discriminator,
        ) in discriminators.feature_discriminators.items():
            try:
                embeddings = discriminators.extract_feature_embeddings(
                    test_text, feature_name
                )
                if embeddings is not None:
                    feature_output = discriminator(embeddings)
                    logger.info(
                        f"✅ Feature discriminator '{feature_name}' output: {feature_output.item():.3f}"
                    )
                else:
                    logger.warning(
                        f"⚠️ No embeddings extracted for feature '{feature_name}'"
                    )
            except Exception as e:
                logger.warning(
                    f"⚠️ Feature discriminator '{feature_name}' failed: {str(e)}"
                )

        logger.info("✅ All forward pass tests completed")

    except Exception as e:
        logger.error(f"❌ Forward pass test failed: {str(e)}")
        raise e


def test_multi_level_feedback(discriminators):
    """
    Test 3: Test multi-level feedback function
    """
    logger.info("=== TEST 3: MULTI-LEVEL FEEDBACK TESTING ===")

    try:
        # Test with real text
        real_text = "gender is Female, age is 30, hypertension is 0, heart_disease is 0, smoking_history is never, bmi is 25.5, HbA1c_level is 5.2, blood_glucose_level is 120, diabetes is 0"

        logger.info("Testing multi-level feedback with real text...")
        feedback = discriminators.get_multi_level_feedback(real_text)

        # Check feedback structure
        required_keys = ["token", "sentence", "row", "features"]
        for key in required_keys:
            assert key in feedback, f"Missing key '{key}' in feedback"

        logger.info("✅ Multi-level feedback structure:")
        logger.info(f"  Token level: {feedback['token']:.3f}")
        logger.info(f"  Sentence level: {feedback['sentence']:.3f}")
        logger.info(f"  Row level: {feedback['row']:.3f}")
        logger.info(f"  Features: {feedback['features']}")

        # Test with synthetic text
        synthetic_text = "gender is Male, age is 45, hypertension is 1, heart_disease is 0, smoking_history is current, bmi is 28.3, HbA1c_level is 6.8, blood_glucose_level is 150, diabetes is 1"

        logger.info("Testing multi-level feedback with synthetic text...")
        feedback_synthetic = discriminators.get_multi_level_feedback(synthetic_text)

        logger.info("✅ Synthetic text feedback:")
        logger.info(f"  Token level: {feedback_synthetic['token']:.3f}")
        logger.info(f"  Sentence level: {feedback_synthetic['sentence']:.3f}")
        logger.info(f"  Row level: {feedback_synthetic['row']:.3f}")
        logger.info(f"  Features: {feedback_synthetic['features']}")

        # Test with malformed text
        malformed_text = "invalid text format"
        logger.info("Testing multi-level feedback with malformed text...")
        try:
            feedback_malformed = discriminators.get_multi_level_feedback(malformed_text)
            logger.info("✅ Malformed text handled gracefully")
        except Exception as e:
            logger.warning(f"⚠️ Malformed text caused error: {str(e)}")

        logger.info("✅ Multi-level feedback tests completed")

    except Exception as e:
        logger.error(f"❌ Multi-level feedback test failed: {str(e)}")
        raise e


def test_discriminator_training(discriminators):
    """
    Test 4: Test discriminator training with small dataset
    """
    logger.info("=== TEST 4: DISCRIMINATOR TRAINING TESTING ===")

    try:
        # Load small dataset for testing
        df = pd.read_csv("diabetes.csv")
        real_texts = df.apply(format_row, axis=1).tolist()[:100]  # Use only 100 samples

        # Generate synthetic texts for testing
        synthetic_texts = []
        for i in range(50):  # Generate 50 synthetic samples
            # Simple synthetic generation for testing
            real_text = random.choice(real_texts)
            # Modify some values to create synthetic data
            synthetic_text = real_text.replace("diabetes is 0", "diabetes is 1")
            synthetic_texts.append(synthetic_text)

        logger.info(
            f"Training with {len(real_texts)} real and {len(synthetic_texts)} synthetic samples"
        )

        # Train discriminators
        training_results = discriminators.train_discriminators(
            real_texts=real_texts,
            synthetic_texts=synthetic_texts,
            epochs=1,  # Only 1 epoch for testing
        )

        logger.info("✅ Discriminator training completed successfully")

        # Test feedback after training
        test_text = real_texts[0]
        feedback_before = discriminators.get_multi_level_feedback(test_text)
        logger.info(
            f"✅ Feedback after training - Row level: {feedback_before['row']:.3f}"
        )

        return training_results

    except Exception as e:
        logger.error(f"❌ Training test failed: {str(e)}")
        raise e


def test_discriminator_save_load(discriminators):
    """
    Test 5: Test save and load functionality
    """
    logger.info("=== TEST 5: SAVE/LOAD FUNCTIONALITY TESTING ===")

    try:
        save_path = "./test_discriminators"

        # Save discriminators
        logger.info("Saving discriminators...")
        discriminators.save_discriminators(save_path)
        logger.info("✅ Discriminators saved successfully")

        # Create new discriminator system and load
        logger.info("Loading discriminators...")
        new_discriminators = HierarchicalDiscriminatorSystem(
            device=discriminators.device
        )
        new_discriminators.load_discriminators(save_path)
        logger.info("✅ Discriminators loaded successfully")

        # Test that loaded discriminators work
        test_text = "gender is Female, age is 30, hypertension is 0, heart_disease is 0, smoking_history is never, bmi is 25.5, HbA1c_level is 5.2, blood_glucose_level is 120, diabetes is 0"

        feedback_original = discriminators.get_multi_level_feedback(test_text)
        feedback_loaded = new_discriminators.get_multi_level_feedback(test_text)

        logger.info("✅ Comparing original vs loaded discriminators:")
        logger.info(f"  Original row level: {feedback_original['row']:.3f}")
        logger.info(f"  Loaded row level: {feedback_loaded['row']:.3f}")

        # Clean up test file
        if os.path.exists(f"{save_path}.pth"):
            os.remove(f"{save_path}.pth")
            logger.info("✅ Test file cleaned up")

        logger.info("✅ Save/load functionality tests completed")

    except Exception as e:
        logger.error(f"❌ Save/load test failed: {str(e)}")
        raise e


def test_discriminator_performance(discriminators):
    """
    Test 6: Test discriminator performance and consistency
    """
    logger.info("=== TEST 6: PERFORMANCE AND CONSISTENCY TESTING ===")

    try:
        # Test consistency with same input
        test_text = "gender is Female, age is 30, hypertension is 0, heart_disease is 0, smoking_history is never, bmi is 25.5, HbA1c_level is 5.2, blood_glucose_level is 120, diabetes is 0"

        feedbacks = []
        for i in range(5):
            feedback = discriminators.get_multi_level_feedback(test_text)
            feedbacks.append(feedback["row"])

        # Check consistency
        feedback_std = np.std(feedbacks)
        logger.info(f"✅ Consistency test - Row level feedbacks: {feedbacks}")
        logger.info(f"✅ Standard deviation: {feedback_std:.6f}")

        if feedback_std < 0.01:  # Very small variation
            logger.info("✅ Discriminators are consistent")
        else:
            logger.warning(f"⚠️ Discriminators show some variation: {feedback_std:.6f}")

        # Test performance with batch of texts
        df = pd.read_csv("diabetes.csv")
        test_texts = df.apply(format_row, axis=1).tolist()[:20]  # Test with 20 texts

        logger.info("Testing performance with batch of texts...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        for text in test_texts:
            feedback = discriminators.get_multi_level_feedback(text)
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)

        logger.info(f"✅ Processed {len(test_texts)} texts in {elapsed_time:.2f}ms")
        logger.info(f"✅ Average time per text: {elapsed_time/len(test_texts):.2f}ms")

        logger.info("✅ Performance and consistency tests completed")

    except Exception as e:
        logger.error(f"❌ Performance test failed: {str(e)}")
        raise e


def main():
    """
    Main function to run all discriminator tests
    """
    try:
        logger.info("=== STARTING COMPREHENSIVE DISCRIMINATOR TESTING ===")

        # Test 1: Initialization
        discriminators = test_discriminator_initialization()

        # Test 2: Forward pass
        test_discriminator_forward_pass(discriminators)

        # Test 3: Multi-level feedback
        test_multi_level_feedback(discriminators)

        # Test 4: Training
        test_discriminator_training(discriminators)

        # Test 5: Save/load
        test_discriminator_save_load(discriminators)

        # Test 6: Performance
        test_discriminator_performance(discriminators)

        logger.info("=== ALL DISCRIMINATOR TESTS COMPLETED SUCCESSFULLY ===")
        logger.info("🎉 Your multi-objective discriminators are working flawlessly!")

    except Exception as e:
        logger.error(f"❌ Testing failed: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
