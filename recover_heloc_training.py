#!/usr/bin/env python3
"""
Recovery script for HELOC discriminator training
Analyzes current state and resumes training from checkpoint
"""

import os
import json
import torch
import logging
from datetime import datetime
from train_heloc_discriminators_wandb import (
    setup_wandb, 
    load_heloc_data, 
    generate_synthetic_heloc_samples,
    train_heloc_discriminators_with_monitoring,
    test_heloc_discriminators
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_current_state():
    """
    Analyze the current state of training and saved models
    """
    logger.info("=== HELOC Training Recovery Analysis ===")
    
    # Check for saved discriminators
    discriminator_path = "./hierarchical_discriminators_heloc.pth"
    if os.path.exists(discriminator_path):
        file_size = os.path.getsize(discriminator_path) / (1024**3)  # GB
        logger.info(f"✅ Found saved discriminators: {discriminator_path} ({file_size:.2f} GB)")
    else:
        logger.info("❌ No saved discriminators found")
    
    # Check for checkpoint directory
    checkpoint_dir = "./hierarchical_discriminators_heloc_checkpoints"
    if os.path.exists(checkpoint_dir):
        logger.info(f"✅ Found checkpoint directory: {checkpoint_dir}")
        
        # Check for checkpoint file
        checkpoint_file = os.path.join(checkpoint_dir, "latest_checkpoint.json")
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"✅ Found checkpoint: epoch {checkpoint_data.get('epoch', 0)}")
                logger.info(f"   Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
                
                # Show metrics
                metrics = checkpoint_data.get('metrics', {})
                for metric_name, values in metrics.items():
                    if values:
                        logger.info(f"   {metric_name}: {values[-1]:.4f} (last value)")
            except Exception as e:
                logger.warning(f"❌ Could not read checkpoint: {e}")
        else:
            logger.info("❌ No checkpoint file found")
    else:
        logger.info("❌ No checkpoint directory found")
    
    # Check for wandb logs
    wandb_dir = "./wandb"
    if os.path.exists(wandb_dir):
        latest_run = os.path.join(wandb_dir, "latest-run")
        if os.path.exists(latest_run):
            logger.info(f"✅ Found wandb logs: {latest_run}")
            
            # Check output log
            output_log = os.path.join(latest_run, "files", "output.log")
            if os.path.exists(output_log):
                with open(output_log, 'r') as f:
                    lines = f.readlines()
                    logger.info(f"   Last {min(5, len(lines))} log lines:")
                    for line in lines[-5:]:
                        logger.info(f"   {line.strip()}")
        else:
            logger.info("❌ No wandb latest run found")
    else:
        logger.info("❌ No wandb directory found")
    
    # Check for synthetic data
    synthetic_data_path = "./output_hierarchical_heloc_clean.csv"
    if os.path.exists(synthetic_data_path):
        import pandas as pd
        df = pd.read_csv(synthetic_data_path)
        logger.info(f"✅ Found synthetic data: {synthetic_data_path} ({len(df)} rows)")
    else:
        logger.info("❌ No synthetic data found")
    
    # Check for PPO models
    ppo_models = [d for d in os.listdir('.') if d.startswith('gpt2_ppo_hierarchical_heloc') and os.path.isdir(d)]
    if ppo_models:
        logger.info(f"✅ Found {len(ppo_models)} PPO models:")
        for model in sorted(ppo_models):
            logger.info(f"   {model}")
    else:
        logger.info("❌ No PPO models found")

def resume_training():
    """
    Resume training from checkpoint or start fresh
    """
    logger.info("\n=== Resuming HELOC Discriminator Training ===")
    
    try:
        # Setup wandb
        setup_wandb()
        
        # Load data
        real_texts = load_heloc_data(max_samples=1000)
        synthetic_texts = generate_synthetic_heloc_samples(real_texts, num_samples=500)
        
        # Resume training
        discriminators, metrics = train_heloc_discriminators_with_monitoring(
            real_texts=real_texts,
            synthetic_texts=synthetic_texts,
            save_path="./hierarchical_discriminators_heloc",
            epochs=5
        )
        
        # Test the discriminators
        test_texts = real_texts[:5] + synthetic_texts[:5]
        test_heloc_discriminators(discriminators, test_texts)
        
        logger.info("✅ Training completed successfully!")
        return discriminators, metrics
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise e

def main():
    """
    Main recovery function
    """
    logger.info("Starting HELOC training recovery...")
    
    # Analyze current state
    analyze_current_state()
    
    # Ask user if they want to resume
    print("\n" + "="*50)
    print("RECOVERY OPTIONS:")
    print("1. Resume training from checkpoint")
    print("2. Start fresh training")
    print("3. Just analyze current state (exit)")
    print("="*50)
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        logger.info("Resuming training from checkpoint...")
        resume_training()
    elif choice == "2":
        logger.info("Starting fresh training...")
        # Remove checkpoint to start fresh
        checkpoint_dir = "./hierarchical_discriminators_heloc_checkpoints"
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info("Removed old checkpoint directory")
        resume_training()
    elif choice == "3":
        logger.info("Analysis complete. Exiting.")
    else:
        logger.error("Invalid choice. Exiting.")

if __name__ == "__main__":
    main() 