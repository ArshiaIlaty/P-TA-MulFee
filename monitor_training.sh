#!/bin/bash

# Script to monitor the training progress

echo "🔍 Training Progress Monitor"
echo "============================"
echo ""

# Check screen sessions
echo "📺 Screen Sessions:"
screen -list | grep discriminator_training
echo ""

# Check for checkpoint files
echo "💾 Checkpoint Files:"
if [ -d "checkpoints" ]; then
    ls -la checkpoints/ 2>/dev/null | head -10
    echo "Latest checkpoint: $(ls -t checkpoints/checkpoint_cycle_*.pth 2>/dev/null | head -1)"
else
    echo "No checkpoints directory found yet"
fi
echo ""

# Check for result files
echo "📊 Result Files:"
for file in training_results_final.json heloc_impact_analysis_final.png; do
    if [ -f "$file" ]; then
        echo "✅ $file exists ($(ls -lh $file | awk '{print $5}'))"
    else
        echo "❌ $file not found yet"
    fi
done
echo ""

# Check wandb logs
echo "📈 Wandb Logs:"
if [ -d "wandb" ]; then
    latest_run=$(ls -t wandb/run-* 2>/dev/null | head -1)
    if [ -n "$latest_run" ]; then
        echo "Latest run: $latest_run"
        echo "Run directory: $(ls -la $latest_run/ 2>/dev/null | wc -l) files"
    else
        echo "No wandb runs found"
    fi
else
    echo "No wandb directory found"
fi
echo ""

# Check system resources
echo "💻 System Resources:"
echo "GPU Usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU not available"
echo ""

echo "Memory Usage:"
free -h | grep -E "Mem|Swap"
echo ""

echo "Disk Usage:"
df -h . | tail -1
echo ""

# Show recent log entries if available
echo "📝 Recent Log Activity:"
if [ -f "wandb/latest-run/logs/debug.log" ]; then
    echo "Last 5 lines from wandb debug log:"
    tail -5 wandb/latest-run/logs/debug.log 2>/dev/null
elif [ -d "wandb" ]; then
    latest_log=$(find wandb/run-* -name "debug.log" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "Last 5 lines from latest wandb log ($latest_log):"
        tail -5 "$latest_log" 2>/dev/null
    fi
else
    echo "No wandb logs found"
fi
echo ""

echo "🎯 To attach to training session:"
echo "   screen -r discriminator_training_20250818_011340"
echo ""
echo "🎯 To kill training session:"
echo "   screen -S discriminator_training_20250818_011340 -X quit"
echo ""
echo "🔄 Run this script again to update:"
echo "   ./monitor_training.sh"
