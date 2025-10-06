#!/usr/bin/env python3
"""
Script to run comparative training between PPO and DPO
"""

import subprocess
import time
import os
import json
from datetime import datetime

def run_training(script_name, args, log_file):
    """Run training script and log output"""
    print(f"\n🚀 Starting {script_name} training...")
    print(f"Command: python {script_name} {' '.join(args)}")
    print(f"Log file: {log_file}")
    
    # Create log directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Run the training script
    cmd = ["python", script_name] + args
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and file
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()
            
            process.wait()
            
        end_time = time.time()
        duration = end_time - start_time
        
        if process.returncode == 0:
            print(f"✅ {script_name} training completed successfully!")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            return True
        else:
            print(f"❌ {script_name} training failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Run comparative training"""
    print("🔬 P-TA PPO vs DPO Comparative Training")
    print("=" * 50)
    
    # Training parameters
    total_steps = 500  # Reduced for comparison
    batch_size = 4
    learning_rate = 1e-5
    checkpoint_interval = 50
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # PPO training
    ppo_args = [
        "--total_steps", str(total_steps),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--checkpoint_interval", str(checkpoint_interval),
        "--save_path", f"./gpt2_ppo_hierarchical_heloc_comparison_{timestamp}",
        "--kl_penalty", "0.1"
    ]
    
    ppo_log = f"comparison_logs/ppo_training_{timestamp}.log"
    
    # DPO training
    dpo_args = [
        "--total_steps", str(total_steps),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--checkpoint_interval", str(checkpoint_interval),
        "--save_path", f"./gpt2_dpo_hierarchical_heloc_comparison_{timestamp}",
        "--beta", "0.1"
    ]
    
    dpo_log = f"comparison_logs/dpo_training_{timestamp}.log"
    
    # Create comparison logs directory
    os.makedirs("comparison_logs", exist_ok=True)
    
    # Save comparison configuration
    config = {
        "timestamp": timestamp,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "checkpoint_interval": checkpoint_interval,
        "ppo_save_path": f"./gpt2_ppo_hierarchical_heloc_comparison_{timestamp}",
        "dpo_save_path": f"./gpt2_dpo_hierarchical_heloc_comparison_{timestamp}",
        "ppo_log": ppo_log,
        "dpo_log": dpo_log
    }
    
    with open(f"comparison_logs/comparison_config_{timestamp}.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Comparison configuration saved to comparison_logs/comparison_config_{timestamp}.json")
    
    # Run PPO training
    print("\n" + "="*50)
    print("🎯 PHASE 1: PPO Training")
    print("="*50)
    
    ppo_success = run_training("ppo_hierarchical_heloc.py", ppo_args, ppo_log)
    
    if not ppo_success:
        print("❌ PPO training failed. Stopping comparison.")
        return
    
    # Run DPO training
    print("\n" + "="*50)
    print("🎯 PHASE 2: DPO Training")
    print("="*50)
    
    dpo_success = run_training("dpo_hierarchical_heloc_simple.py", dpo_args, dpo_log)
    
    if not dpo_success:
        print("❌ DPO training failed.")
    
    # Summary
    print("\n" + "="*50)
    print("📊 COMPARISON SUMMARY")
    print("="*50)
    
    if ppo_success and dpo_success:
        print("✅ Both PPO and DPO training completed successfully!")
        print(f"📁 PPO model saved to: {config['ppo_save_path']}")
        print(f"📁 DPO model saved to: {config['dpo_save_path']}")
        print(f"📄 PPO log: {config['ppo_log']}")
        print(f"📄 DPO log: {config['dpo_log']}")
        
        print("\n🔍 Next steps for analysis:")
        print("1. Compare training curves in the log files")
        print("2. Analyze final model performance")
        print("3. Generate synthetic data with both models")
        print("4. Evaluate data quality, diversity, and utility")
        print("5. Check the comparison configuration file for details")
        
    elif ppo_success:
        print("⚠️  PPO training completed, but DPO training failed.")
        print("You can still analyze the PPO results.")
        
    else:
        print("❌ Both training runs failed. Check the log files for details.")
    
    print(f"\n📋 Configuration saved to: comparison_logs/comparison_config_{timestamp}.json")

if __name__ == "__main__":
    main() 