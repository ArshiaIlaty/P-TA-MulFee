#!/bin/bash

# Simple script to run HELOC discriminator training in background
echo "🚀 Starting HELOC discriminator training in background..."

# Create logs directory
mkdir -p training_logs

# Start training with nohup (will continue even if terminal is closed)
nohup python train_heloc_discriminators_wandb.py > training_logs/heloc_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get the process ID
TRAINING_PID=$!
echo "✅ Training started with PID: $TRAINING_PID"
echo "📝 Log file: training_logs/heloc_training_$(date +%Y%m%d_%H%M%S).log"

# Save PID for monitoring
echo $TRAINING_PID > training_logs/training.pid

echo ""
echo "🎯 You can now safely put your laptop to sleep!"
echo ""
echo "📊 To monitor progress:"
echo "   tail -f training_logs/heloc_training_*.log"
echo ""
echo "🛑 To stop training:"
echo "   kill $(cat training_logs/training.pid)"
echo ""
echo "📈 To check if training is running:"
echo "   ps aux | grep train_heloc_discriminators_wandb.py" 