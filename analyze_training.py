#!/usr/bin/env python3
"""
Training Analysis Script for PPO Hierarchical HELOC
Analyzes training logs and provides recommendations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_training_logs(log_file="ppo_training_logs/training_summary.jsonl"):
    """Load and parse training logs"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found!")
        return None
    
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return data

def analyze_training_progress(data):
    """Analyze training progress and trends"""
    if not data:
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Calculate moving averages for smoother trend analysis
    window_size = min(10, len(df) // 4)  # Adaptive window size
    df['reward_ma'] = df['mean_reward'].rolling(window=window_size, center=True).mean()
    
    # Calculate overall statistics
    stats = {
        'total_steps': len(df),
        'final_reward': df['mean_reward'].iloc[-1],
        'best_reward': df['mean_reward'].max(),
        'worst_reward': df['mean_reward'].min(),
        'avg_reward': df['mean_reward'].mean(),
        'reward_std': df['mean_reward'].std(),
        'reward_trend': 'increasing' if df['mean_reward'].iloc[-1] > df['mean_reward'].iloc[0] else 'decreasing',
        'convergence_stable': df['mean_reward'].iloc[-window_size:].std() < 0.02,  # Low variance in recent steps
        'reward_components': {
            'token': df['avg_reward_components'].apply(lambda x: x['token']).mean(),
            'sentence': df['avg_reward_components'].apply(lambda x: x['sentence']).mean(),
            'row': df['avg_reward_components'].apply(lambda x: x['row']).mean(),
            'features_mean': df['avg_reward_components'].apply(lambda x: x['features_mean']).mean()
        }
    }
    
    # Analyze recent performance (last 25% of training)
    recent_start = max(0, len(df) - len(df) // 4)
    recent_data = df.iloc[recent_start:]
    stats['recent_avg'] = recent_data['mean_reward'].mean()
    stats['recent_std'] = recent_data['mean_reward'].std()
    stats['recent_trend'] = 'stable' if stats['recent_std'] < 0.02 else 'unstable'
    
    return df, stats

def plot_training_curves(df, save_path="training_analysis.png"):
    """Create visualization of training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Main reward curve
    axes[0, 0].plot(df.index, df['mean_reward'], alpha=0.6, label='Step Reward')
    if 'reward_ma' in df.columns:
        axes[0, 0].plot(df.index, df['reward_ma'], 'r-', linewidth=2, label=f'Moving Average (window={len(df)//4})')
    axes[0, 0].set_title('Training Reward Over Time')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward components
    components = ['token', 'sentence', 'row', 'features_mean']
    for comp in components:
        comp_values = df['avg_reward_components'].apply(lambda x: x[comp])
        axes[0, 1].plot(df.index, comp_values, label=comp.capitalize(), alpha=0.7)
    axes[0, 1].set_title('Reward Components Over Time')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Component Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 0].hist(df['mean_reward'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['mean_reward'].mean(), color='red', linestyle='--', label=f'Mean: {df["mean_reward"].mean():.3f}')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Mean Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # GPU Memory usage
    axes[1, 1].plot(df.index, df['gpu_memory_allocated_gb'], label='Allocated', alpha=0.7)
    axes[1, 1].plot(df.index, df['gpu_memory_reserved_gb'], label='Reserved', alpha=0.7)
    axes[1, 1].set_title('GPU Memory Usage')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Memory (GB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training analysis plot saved to {save_path}")
    plt.close()

def generate_recommendations(stats, target_steps=5000):
    """Generate recommendations based on training analysis"""
    recommendations = []
    
    # Analyze current performance
    current_reward = stats['final_reward']
    best_reward = stats['best_reward']
    recent_avg = stats['recent_avg']
    
    # Check if training is still improving
    if current_reward < best_reward * 0.95:  # Current reward is significantly below best
        recommendations.append("⚠️  Current performance is below peak performance. Training may have regressed.")
    
    # Check convergence
    if stats['convergence_stable'] and stats['recent_trend'] == 'stable':
        if recent_avg < 0.55:  # Low performance threshold
            recommendations.append("🔴 Training has converged to a low performance level. Consider:")
            recommendations.append("   - Reducing learning rate")
            recommendations.append("   - Adjusting reward function weights")
            recommendations.append("   - Increasing KL penalty for more exploration")
        else:
            recommendations.append("🟢 Training has converged to a good performance level.")
    else:
        recommendations.append("🟡 Training has not yet converged. Consider continuing training.")
    
    # Check reward components balance
    components = stats['reward_components']
    component_std = np.std(list(components.values()))
    if component_std > 0.1:  # High variance in component rewards
        recommendations.append("⚠️  Reward components are imbalanced. Consider adjusting weights in reward function.")
    
    # Specific recommendations based on current step count
    current_steps = stats['total_steps']
    if current_steps < target_steps:
        if stats['reward_trend'] == 'increasing' or not stats['convergence_stable']:
            recommendations.append(f"✅ Continue training to {target_steps} steps - performance is still improving.")
        else:
            recommendations.append(f"⚠️  Consider continuing training but monitor for improvement.")
    
    # Learning rate recommendations
    if stats['recent_std'] > 0.05:  # High variance in recent performance
        recommendations.append("💡 Consider reducing learning rate for more stable training.")
    
    return recommendations

def main():
    print("=== PPO Training Analysis ===\n")
    
    # Load training data
    data = load_training_logs()
    if not data:
        print("No training data found!")
        return
    
    # Analyze training progress
    df, stats = analyze_training_progress(data)
    
    # Print analysis results
    print(f"Training Summary:")
    print(f"  Total Steps: {stats['total_steps']}")
    print(f"  Final Reward: {stats['final_reward']:.4f}")
    print(f"  Best Reward: {stats['best_reward']:.4f}")
    print(f"  Average Reward: {stats['avg_reward']:.4f}")
    print(f"  Reward Trend: {stats['reward_trend']}")
    print(f"  Recent Average: {stats['recent_avg']:.4f}")
    print(f"  Recent Stability: {stats['recent_trend']}")
    print(f"  Convergence Stable: {stats['convergence_stable']}")
    
    print(f"\nReward Components (Average):")
    for comp, value in stats['reward_components'].items():
        print(f"  {comp.capitalize()}: {value:.4f}")
    
    # Generate and print recommendations
    print(f"\n=== Recommendations ===")
    recommendations = generate_recommendations(stats)
    for rec in recommendations:
        print(rec)
    
    # Create visualization
    try:
        plot_training_curves(df)
        print(f"\n📊 Training analysis plot created successfully!")
    except Exception as e:
        print(f"⚠️  Failed to create plot: {e}")
    
    # Final assessment
    print(f"\n=== Final Assessment ===")
    if stats['final_reward'] > 0.55:
        print("🟢 Good performance achieved!")
    elif stats['final_reward'] > 0.45:
        print("🟡 Moderate performance - room for improvement")
    else:
        print("🔴 Low performance - significant improvements needed")
    
    if stats['total_steps'] < 5000:
        print(f"📈 Training stopped at {stats['total_steps']}/5000 steps")
        if stats['reward_trend'] == 'increasing' or not stats['convergence_stable']:
            print("   → Consider continuing training")
        else:
            print("   → Training may have plateaued")

if __name__ == "__main__":
    main() 