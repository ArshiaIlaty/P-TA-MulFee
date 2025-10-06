#!/bin/bash

# Script to run the improved discriminator training in a screen session
# This prevents crashes from disconnections and allows monitoring

echo "Starting improved discriminator training in screen session..."

# Create a unique screen session name with timestamp
SESSION_NAME="discriminator_training_$(date +%Y%m%d_%H%M%S)"

# Create the screen session and run the training
screen -dmS "$SESSION_NAME" bash -c "
    echo 'Starting training in screen session: $SESSION_NAME'
    echo 'Timestamp: $(date)'
    echo 'Working directory: $(pwd)'
    echo 'Python script: improved_discriminator_training_final.py'
    echo ''
    echo 'Training will start in 5 seconds...'
    sleep 5
    
    # Activate conda environment if needed
    # conda activate your_env_name
    
    # Run the training script
    python improved_discriminator_training_simple_fixed.py
    
    echo ''
    echo 'Training completed at: $(date)'
    echo 'Press any key to close this screen session...'
    read -n 1
"

# Wait a moment for screen to start
sleep 2

# Check if screen session was created successfully
if screen -list | grep -q "$SESSION_NAME"; then
    echo "✅ Training started successfully in screen session: $SESSION_NAME"
    echo ""
    echo "📋 Useful commands:"
    echo "  - Attach to session: screen -r $SESSION_NAME"
    echo "  - List sessions: screen -list"
    echo "  - Detach from session: Ctrl+A, then D"
    echo "  - Kill session: screen -S $SESSION_NAME -X quit"
    echo ""
    echo "🔍 Monitor training progress:"
    echo "  - Check wandb dashboard for real-time metrics"
    echo "  - Monitor checkpoint files in ./checkpoints/"
    echo "  - Check logs in wandb/ directory"
    echo ""
    echo "📁 Output files will be saved to:"
    echo "  - training_results_final.json"
    echo "  - heloc_impact_analysis_final.png"
    echo "  - checkpoints/ directory"
    echo ""
    echo "🚀 To attach to the training session now, run:"
    echo "   screen -r $SESSION_NAME"
else
    echo "❌ Failed to start screen session"
    exit 1
fi
