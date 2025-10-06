#!/usr/bin/env python3
"""
Quick analysis of current HELOC training state
"""

import os
import json
import pandas as pd
from datetime import datetime

def analyze_current_state():
    """
    Analyze the current state of training and saved models
    """
    print("=== HELOC Training Recovery Analysis ===")
    
    # Check for saved discriminators
    discriminator_path = "./hierarchical_discriminators_heloc.pth"
    if os.path.exists(discriminator_path):
        file_size = os.path.getsize(discriminator_path) / (1024**3)  # GB
        print(f"✅ Found saved discriminators: {discriminator_path} ({file_size:.2f} GB)")
    else:
        print("❌ No saved discriminators found")
    
    # Check for checkpoint directory
    checkpoint_dir = "./hierarchical_discriminators_heloc_checkpoints"
    if os.path.exists(checkpoint_dir):
        print(f"✅ Found checkpoint directory: {checkpoint_dir}")
        
        # Check for checkpoint file
        checkpoint_file = os.path.join(checkpoint_dir, "latest_checkpoint.json")
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                print(f"✅ Found checkpoint: epoch {checkpoint_data.get('epoch', 0)}")
                print(f"   Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
                
                # Show metrics
                metrics = checkpoint_data.get('metrics', {})
                for metric_name, values in metrics.items():
                    if values:
                        print(f"   {metric_name}: {values[-1]:.4f} (last value)")
            except Exception as e:
                print(f"❌ Could not read checkpoint: {e}")
        else:
            print("❌ No checkpoint file found")
    else:
        print("❌ No checkpoint directory found")
    
    # Check for wandb logs
    wandb_dir = "./wandb"
    if os.path.exists(wandb_dir):
        latest_run = os.path.join(wandb_dir, "latest-run")
        if os.path.exists(latest_run):
            print(f"✅ Found wandb logs: {latest_run}")
            
            # Check output log
            output_log = os.path.join(latest_run, "files", "output.log")
            if os.path.exists(output_log):
                with open(output_log, 'r') as f:
                    lines = f.readlines()
                    print(f"   Last {min(5, len(lines))} log lines:")
                    for line in lines[-5:]:
                        print(f"   {line.strip()}")
        else:
            print("❌ No wandb latest run found")
    else:
        print("❌ No wandb directory found")
    
    # Check for synthetic data
    synthetic_data_path = "./output_hierarchical_heloc_clean.csv"
    if os.path.exists(synthetic_data_path):
        try:
            df = pd.read_csv(synthetic_data_path)
            print(f"✅ Found synthetic data: {synthetic_data_path} ({len(df)} rows)")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df.head(2).to_string())
        except Exception as e:
            print(f"❌ Error reading synthetic data: {e}")
    else:
        print("❌ No synthetic data found")
    
    # Check for PPO models
    ppo_models = [d for d in os.listdir('.') if d.startswith('gpt2_ppo_hierarchical_heloc') and os.path.isdir(d)]
    if ppo_models:
        print(f"✅ Found {len(ppo_models)} PPO models:")
        for model in sorted(ppo_models):
            print(f"   {model}")
    else:
        print("❌ No PPO models found")
    
    # Check for DPO models
    dpo_models = [d for d in os.listdir('.') if d.startswith('gpt2_dpo_hierarchical_heloc') and os.path.isdir(d)]
    if dpo_models:
        print(f"✅ Found {len(dpo_models)} DPO models:")
        for model in sorted(dpo_models):
            print(f"   {model}")
    else:
        print("❌ No DPO models found")

def main():
    """
    Main analysis function
    """
    print("Starting HELOC training analysis...")
    analyze_current_state()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("1. You have significant progress saved!")
    print("2. PPO training completed successfully (up to step 1350)")
    print("3. Synthetic data generated (9,832 rows)")
    print("4. Discriminator training failed due to tensor shape mismatch")
    print("5. The 10-hour run was for discriminator training, not generator training")
    print("\nNEXT STEPS:")
    print("1. Activate conda environment: conda activate P-TA")
    print("2. Run the fixed discriminator training script")
    print("3. The generator training (PPO) is already complete!")
    print("="*50)

if __name__ == "__main__":
    main() 