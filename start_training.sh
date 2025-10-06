#!/bin/bash

echo "🚀 Starting HELOC discriminator training..."

# Create logs directory
mkdir -p training_logs

# Start training in background
nohup python train_heloc_discriminators_wandb.py > training_logs/heloc_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get PID
TRAINING_PID=$!
echo "✅ Training started with PID: $TRAINING_PID"

# Save PID
echo $TRAINING_PID > training_logs/training.pid

echo ""
echo "🎯 You can now safely put your laptop to sleep!"
echo ""
echo "📊 Monitor progress: tail -f training_logs/heloc_training_*.log"
echo "🛑 Stop training: kill $(cat training_logs/training.pid)" 