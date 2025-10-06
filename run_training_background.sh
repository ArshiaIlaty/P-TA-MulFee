#!/bin/bash

# Background training script for HELOC discriminators
# This script runs the training in the background with proper logging

echo "🚀 Starting HELOC discriminator training in background..."
echo "📅 Started at: $(date)"
echo "💻 Process ID: $$"

# Create logs directory
mkdir -p training_logs

# Set up log file with timestamp
LOG_FILE="training_logs/heloc_discriminator_training_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="training_logs/training.pid"

# Save PID for monitoring
echo $$ > $PID_FILE

echo "📝 Log file: $LOG_FILE"
echo "🆔 PID file: $PID_FILE"

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    rm -f $PID_FILE
    echo "✅ Training completed at: $(date)"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Activate conda environment and run training
echo "🔧 Activating P-TA conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate P-TA

echo "🚀 Starting discriminator training..."
echo "⏰ Training started at: $(date)" | tee -a $LOG_FILE

# Run the training script with full logging
python train_heloc_discriminators_wandb.py 2>&1 | tee -a $LOG_FILE

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
echo "🏁 Training finished with exit code: $EXIT_CODE" | tee -a $LOG_FILE

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!" | tee -a $LOG_FILE
else
    echo "❌ Training failed with exit code: $EXIT_CODE" | tee -a $LOG_FILE
fi

echo "📊 Final log saved to: $LOG_FILE"
echo "🎯 You can monitor progress with: tail -f $LOG_FILE" 