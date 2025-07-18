import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_comparison_visualizations():
    """Create comprehensive visualizations for the comparison"""
    
    # Load the summary data
    df = pd.read_csv('hierarchical_vs_normal_summary.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hierarchical vs Normal GPT-2 Approach Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df['Model']))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['Baseline_Accuracy'], width, label='Real Data (Baseline)', alpha=0.8)
    bars2 = ax1.bar(x + width/2, df['Hierarchical_Accuracy'], width, label='Hierarchical Synthetic', alpha=0.8)
    bars3 = ax1.bar(x + width*1.5, df['Normal_Accuracy'], width, label='Normal Synthetic', alpha=0.8)
    
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison Across Approaches')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(df['Model'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Utility Preservation
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, df['Hierarchical_Utility_Preservation'], width, 
                    label='Hierarchical', color='orange', alpha=0.8)
    bars2 = ax2.bar(x + width/2, df['Normal_Utility_Preservation'], width, 
                    label='Normal', color='green', alpha=0.8)
    
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Utility Preservation (%)')
    ax2.set_title('Utility Preservation Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Model'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. F1 Score Comparison
    ax3 = axes[1, 0]
    bars1 = ax3.bar(x - width/2, df['Baseline_F1'], width, label='Real Data (Baseline)', alpha=0.8)
    bars2 = ax3.bar(x + width/2, df['Hierarchical_F1'], width, label='Hierarchical Synthetic', alpha=0.8)
    bars3 = ax3.bar(x + width*1.5, df['Normal_F1'], width, label='Normal Synthetic', alpha=0.8)
    
    ax3.set_xlabel('Model Type')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Comparison Across Approaches')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(df['Model'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Direct Improvement Comparison
    ax4 = axes[1, 1]
    colors = ['red' if x < 0 else 'green' for x in df['Hierarchical_vs_Normal_Improvement']]
    bars = ax4.bar(df['Model'], df['Hierarchical_vs_Normal_Improvement'], color=colors, alpha=0.8)
    
    ax4.set_xlabel('Model Type')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Hierarchical vs Normal Improvement')
    ax4.set_xticklabels(df['Model'], rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('hierarchical_vs_normal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Baseline_Accuracy']:.3f}",
            f"{row['Hierarchical_Accuracy']:.3f}",
            f"{row['Normal_Accuracy']:.3f}",
            f"{row['Hierarchical_Utility_Preservation']:.1f}%",
            f"{row['Normal_Utility_Preservation']:.1f}%",
            f"{row['Hierarchical_vs_Normal_Improvement']:+.1f}%"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Baseline', 'Hierarchical', 'Normal', 
                              'Hierarchical Utility', 'Normal Utility', 'Improvement'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.12, 0.12, 0.12, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the improvement column
    for i in range(1, len(table_data) + 1):
        improvement = float(table_data[i-1][-1].replace('%', ''))
        if improvement > 0:
            table[(i, 6)].set_facecolor('#90EE90')  # Light green
        else:
            table[(i, 6)].set_facecolor('#FFB6C1')  # Light red
    
    plt.title('Comprehensive Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('comparison_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_quality_analysis():
    """Analyze data quality differences"""
    
    # Load the datasets
    real_data = pd.read_csv('diabetes.csv')
    hierarchical_data = pd.read_csv('output_hierarchical_clean.csv')
    normal_data = pd.read_csv('output_diabetes_clean.csv')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Analysis', fontsize=16, fontweight='bold')
    
    # 1. Dataset sizes
    ax1 = axes[0, 0]
    sizes = [len(real_data), len(hierarchical_data), len(normal_data)]
    labels = ['Real Data', 'Hierarchical\nSynthetic', 'Normal\nSynthetic']
    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(labels, sizes, color=colors, alpha=0.8)
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Dataset Sizes')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature distribution comparison (for numerical features)
    ax2 = axes[0, 1]
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    x = np.arange(len(numerical_cols))
    width = 0.25
    
    real_means = [real_data[col].mean() for col in numerical_cols]
    hierarchical_means = [hierarchical_data[col].mean() for col in numerical_cols]
    normal_means = [normal_data[col].mean() for col in numerical_cols]
    
    ax2.bar(x - width, real_means, width, label='Real Data', alpha=0.8)
    ax2.bar(x, hierarchical_means, width, label='Hierarchical', alpha=0.8)
    ax2.bar(x + width, normal_means, width, label='Normal', alpha=0.8)
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Mean Value')
    ax2.set_title('Feature Distribution Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(numerical_cols, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Target distribution
    ax3 = axes[1, 0]
    
    real_target_dist = real_data['diabetes'].value_counts()
    hierarchical_target_dist = hierarchical_data['diabetes'].value_counts()
    normal_target_dist = normal_data['diabetes'].value_counts()
    
    x = np.arange(2)
    width = 0.25
    
    ax3.bar(x - width, [real_target_dist[0], real_target_dist[1]], width, label='Real Data', alpha=0.8)
    ax3.bar(x, [hierarchical_target_dist[0], hierarchical_target_dist[1]], width, label='Hierarchical', alpha=0.8)
    ax3.bar(x + width, [normal_target_dist[0], normal_target_dist[1]], width, label='Normal', alpha=0.8)
    
    ax3.set_xlabel('Diabetes Status')
    ax3.set_ylabel('Count')
    ax3.set_title('Target Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Data retention rate
    ax4 = axes[1, 1]
    
    original_generated = 100000  # Based on logs
    hierarchical_retention = len(hierarchical_data) / original_generated * 100
    normal_retention = len(normal_data) / original_generated * 100
    
    retention_data = [hierarchical_retention, normal_retention]
    labels = ['Hierarchical\nSynthetic', 'Normal\nSynthetic']
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax4.bar(labels, retention_data, color=colors, alpha=0.8)
    ax4.set_ylabel('Retention Rate (%)')
    ax4.set_title('Data Quality Retention Rate')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating comparison visualizations...")
    create_comparison_visualizations()
    
    print("Creating data quality analysis...")
    create_data_quality_analysis()
    
    print("Visualizations saved as:")
    print("- hierarchical_vs_normal_comparison.png")
    print("- comparison_summary_table.png") 
    print("- data_quality_analysis.png") 