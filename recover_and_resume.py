#!/usr/bin/env python3
"""
Comprehensive recovery and resume script for HELOC training
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime

def check_conda_environment():
    """Check if we're in the right conda environment"""
    print("🔍 Checking conda environment...")
    
    # Check if we're in the P-TA environment
    if 'P-TA' not in os.environ.get('CONDA_DEFAULT_ENV', ''):
        print("⚠️  Not in P-TA conda environment!")
        print("   Please run: conda activate P-TA")
        return False
    
    print("✅ P-TA conda environment detected")
    return True

def backup_existing_files():
    """Backup existing files before recovery"""
    print("📦 Creating backup of existing files...")
    
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup existing discriminators
    if os.path.exists("./hierarchical_discriminators_heloc.pth"):
        shutil.copy2("./hierarchical_discriminators_heloc.pth", backup_dir)
        print(f"   ✅ Backed up discriminators to {backup_dir}")
    
    # Backup checkpoint directory if it exists
    if os.path.exists("./hierarchical_discriminators_heloc_checkpoints"):
        shutil.copytree("./hierarchical_discriminators_heloc_checkpoints", 
                       os.path.join(backup_dir, "hierarchical_discriminators_heloc_checkpoints"))
        print(f"   ✅ Backed up checkpoints to {backup_dir}")
    
    return backup_dir

def analyze_training_state():
    """Analyze current training state"""
    print("📊 Analyzing current training state...")
    
    # Check for saved discriminators
    discriminator_path = "./hierarchical_discriminators_heloc.pth"
    if os.path.exists(discriminator_path):
        file_size = os.path.getsize(discriminator_path) / (1024**3)  # GB
        print(f"   ✅ Found discriminators: {file_size:.2f} GB")
        
        # Check if it's a valid file
        try:
            import torch
            state_dict = torch.load(discriminator_path, map_location='cpu')
            print(f"   ✅ Discriminator file is valid (contains {len(state_dict)} keys)")
        except Exception as e:
            print(f"   ❌ Discriminator file is corrupted: {e}")
            return False
    else:
        print("   ❌ No discriminators found")
        return False
    
    # Check for synthetic data
    synthetic_data_path = "./output_hierarchical_heloc_clean.csv"
    if os.path.exists(synthetic_data_path):
        import pandas as pd
        df = pd.read_csv(synthetic_data_path)
        print(f"   ✅ Found synthetic data: {len(df)} rows")
    else:
        print("   ❌ No synthetic data found")
        return False
    
    return True

def run_discriminator_training():
    """Run the fixed discriminator training script"""
    print("🚀 Starting discriminator training with checkpointing...")
    
    try:
        # Run the fixed training script
        result = subprocess.run([
            sys.executable, "train_heloc_discriminators_wandb.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Discriminator training completed successfully!")
            return True
        else:
            print(f"❌ Discriminator training failed:")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

def test_discriminators():
    """Test the trained discriminators"""
    print("🧪 Testing discriminators...")
    
    try:
        # Create a simple test script
        test_script = """
import torch
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
import pandas as pd

# Load discriminators
discriminators = HierarchicalDiscriminatorSystem(dataset_type="heloc")
discriminators.load_discriminators("./hierarchical_discriminators_heloc")

# Load some test data
df = pd.read_csv("./output_hierarchical_heloc_clean.csv")
test_text = " ".join([f"{col}:{val}" for col, val in df.iloc[0].items()])

# Test discriminator
feedback = discriminators.get_multi_level_feedback(test_text)
print(f"Token feedback: {feedback['token']:.3f}")
print(f"Sentence feedback: {feedback['sentence']:.3f}")
print(f"Row feedback: {feedback['row']:.3f}")
print(f"Feature feedback: {feedback['features']}")

# Calculate overall quality score
quality_score = (
    feedback["token"] * 0.2
    + feedback["sentence"] * 0.3
    + feedback["row"] * 0.3
    + sum(feedback["features"].values()) / len(feedback["features"]) * 0.2
)
print(f"Overall quality score: {quality_score:.3f}")
"""
        
        with open("test_discriminators_temp.py", "w") as f:
            f.write(test_script)
        
        result = subprocess.run([
            sys.executable, "test_discriminators_temp.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Discriminator test successful!")
            print(f"   Output: {result.stdout}")
            return True
        else:
            print(f"❌ Discriminator test failed:")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing discriminators: {e}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists("test_discriminators_temp.py"):
            os.remove("test_discriminators_temp.py")

def generate_new_synthetic_data():
    """Generate new synthetic data using the trained discriminators"""
    print("🔄 Generating new synthetic data with discriminators...")
    
    try:
        result = subprocess.run([
            sys.executable, "generate_hierarchical_heloc.py"
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
        
        if result.returncode == 0:
            print("✅ New synthetic data generated successfully!")
            return True
        else:
            print(f"❌ Data generation failed:")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Data generation timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False

def main():
    """Main recovery function"""
    print("="*60)
    print("🔄 HELOC TRAINING RECOVERY AND RESUME")
    print("="*60)
    
    # Step 1: Check environment
    if not check_conda_environment():
        print("\n❌ Please activate the P-TA conda environment first:")
        print("   conda activate P-TA")
        return
    
    # Step 2: Analyze current state
    if not analyze_training_state():
        print("\n❌ Current state analysis failed. Cannot proceed.")
        return
    
    # Step 3: Create backup
    backup_dir = backup_existing_files()
    
    # Step 4: Recovery options
    print("\n" + "="*50)
    print("RECOVERY OPTIONS:")
    print("1. Resume discriminator training (recommended)")
    print("2. Test existing discriminators")
    print("3. Generate new synthetic data")
    print("4. Full recovery (train + test + generate)")
    print("5. Exit")
    print("="*50)
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🔄 Resuming discriminator training...")
        if run_discriminator_training():
            print("✅ Recovery successful!")
        else:
            print("❌ Recovery failed!")
    
    elif choice == "2":
        print("\n🧪 Testing discriminators...")
        if test_discriminators():
            print("✅ Testing successful!")
        else:
            print("❌ Testing failed!")
    
    elif choice == "3":
        print("\n🔄 Generating new synthetic data...")
        if generate_new_synthetic_data():
            print("✅ Data generation successful!")
        else:
            print("❌ Data generation failed!")
    
    elif choice == "4":
        print("\n🔄 Full recovery process...")
        success = True
        
        print("   Step 1: Training discriminators...")
        if not run_discriminator_training():
            success = False
            print("   ❌ Training failed!")
        
        if success:
            print("   Step 2: Testing discriminators...")
            if not test_discriminators():
                success = False
                print("   ❌ Testing failed!")
        
        if success:
            print("   Step 3: Generating new data...")
            if not generate_new_synthetic_data():
                success = False
                print("   ❌ Data generation failed!")
        
        if success:
            print("✅ Full recovery completed successfully!")
        else:
            print("❌ Full recovery failed!")
    
    elif choice == "5":
        print("👋 Exiting recovery process.")
        return
    
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    print(f"\n📦 Backup created in: {backup_dir}")
    print("🎯 Recovery process completed!")

if __name__ == "__main__":
    main() 