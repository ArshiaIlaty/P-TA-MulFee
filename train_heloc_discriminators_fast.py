import torch
import pandas as pd
import numpy as np
import logging
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
from utils import format_row
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_heloc_data(csv_file="heloc.csv", max_samples=1000):
    """
    Load and format HELOC data for discriminator training (limited samples for speed)
    """
    logger.info(f"Loading HELOC data from {csv_file}")
    
    df = pd.read_csv(csv_file)
    # Use only a subset for faster training
    df = df.head(max_samples)
    logger.info(f"Using {len(df)} HELOC samples for training")
    
    # Format data as text for discriminator training
    real_texts = df.apply(format_row, axis=1).tolist()
    logger.info(f"Formatted {len(real_texts)} HELOC text samples")
    
    return real_texts

def generate_synthetic_heloc_samples(real_texts, num_samples=500):
    """
    Generate synthetic HELOC samples for discriminator training (simplified)
    """
    logger.info(f"Generating {num_samples} synthetic HELOC samples for training")
    
    synthetic_texts = []
    
    for i in range(num_samples):
        # Take a real sample and make small modifications
        base_text = real_texts[i % len(real_texts)]
        
        # Simple modification: replace some values
        modified_text = base_text.replace("Good", "Bad") if "Good" in base_text else base_text.replace("Bad", "Good")
        
        # Add some random noise to numeric values
        import re
        def replace_numbers(match):
            num = float(match.group())
            # Add small random noise (±10%)
            noise = np.random.uniform(-0.1, 0.1)
            new_num = num * (1 + noise)
            return str(int(new_num) if new_num.is_integer() else round(new_num, 1))
        
        modified_text = re.sub(r'\d+\.?\d*', replace_numbers, modified_text)
        synthetic_texts.append(modified_text)
    
    logger.info(f"Generated {len(synthetic_texts)} synthetic HELOC samples")
    return synthetic_texts

def train_heloc_discriminators_fast(real_texts, synthetic_texts, save_path="./hierarchical_discriminators_heloc"):
    """
    Train hierarchical discriminators for HELOC data (fast version)
    """
    logger.info("Training hierarchical HELOC discriminators (fast version)...")
    
    # Initialize discriminators
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discriminators = HierarchicalDiscriminatorSystem(device=device, dataset_type="heloc")
    
    # Train discriminators with fewer epochs
    discriminators.train_discriminators(real_texts, synthetic_texts, epochs=1)
    
    # Save discriminators
    discriminators.save_discriminators(save_path)
    
    logger.info(f"HELOC discriminators trained and saved to {save_path}")
    return discriminators

def test_heloc_discriminators(discriminators, test_texts):
    """
    Test the trained discriminators on some sample texts
    """
    logger.info("Testing HELOC discriminators...")
    
    for i, text in enumerate(test_texts[:3]):  # Test first 3 samples
        feedback = discriminators.get_multi_level_feedback(text)
        
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Text: {text[:100]}...")
        logger.info(f"  Token feedback: {feedback['token']:.3f}")
        logger.info(f"  Sentence feedback: {feedback['sentence']:.3f}")
        logger.info(f"  Row feedback: {feedback['row']:.3f}")
        logger.info(f"  Feature feedback: {feedback['features']}")
        
        # Calculate overall quality score
        quality_score = (
            feedback["token"] * 0.2
            + feedback["sentence"] * 0.3
            + feedback["row"] * 0.3
            + np.mean(list(feedback["features"].values())) * 0.2
        )
        logger.info(f"  Overall quality score: {quality_score:.3f}")

def main():
    """
    Main function to train HELOC discriminators (fast version)
    """
    logger.info("=== TRAINING HELOC HIERARCHICAL DISCRIMINATORS (FAST VERSION) ===")
    
    try:
        # Load real HELOC data (limited samples)
        real_texts = load_heloc_data(max_samples=500)
        
        # Generate synthetic samples for training (limited samples)
        synthetic_texts = generate_synthetic_heloc_samples(real_texts, num_samples=500)
        
        # Train discriminators (fast version)
        discriminators = train_heloc_discriminators_fast(real_texts, synthetic_texts)
        
        # Test discriminators
        test_heloc_discriminators(discriminators, real_texts[:5])
        
        logger.info("=== HELOC DISCRIMINATOR TRAINING COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Error training HELOC discriminators: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 