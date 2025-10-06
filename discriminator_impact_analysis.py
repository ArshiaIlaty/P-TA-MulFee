import torch
import pandas as pd
import numpy as np
import logging
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
from utils import format_row
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscriminatorImpactAnalyzer:
    """
    Comprehensive analyzer to understand discriminator impact on data quality
    """
    
    def __init__(self, discriminators, device="cuda"):
        self.discriminators = discriminators
        self.device = device
        self.impact_metrics = defaultdict(list)
        
    def analyze_single_sample(self, text, sample_id=0):
        """
        Analyze the impact of each discriminator on a single sample
        """
        logger.info(f"=== ANALYZING SAMPLE {sample_id} ===")
        logger.info(f"Text: {text[:150]}...")
        
        # Get individual discriminator feedback
        feedback = self.discriminators.get_multi_level_feedback(text)
        
        # Calculate individual impacts
        impacts = {
            'token': feedback['token'],
            'sentence': feedback['sentence'], 
            'row': feedback['row'],
            'feature_avg': np.mean(list(feedback['features'].values())),
            'feature_risk_estimate': feedback['features']['risk_estimate'],
            'feature_trade_history': feedback['features']['trade_history'],
            'feature_delinquency': feedback['features']['delinquency'],
            'feature_credit_utilization': feedback['features']['credit_utilization']
        }
        
        # Calculate weighted quality score (standard formula)
        weighted_score = (
            impacts['token'] * 0.2 +
            impacts['sentence'] * 0.3 +
            impacts['row'] * 0.3 +
            impacts['feature_avg'] * 0.2
        )
        
        # Calculate individual contributions
        contributions = {
            'token_contribution': impacts['token'] * 0.2,
            'sentence_contribution': impacts['sentence'] * 0.3,
            'row_contribution': impacts['row'] * 0.3,
            'feature_contribution': impacts['feature_avg'] * 0.2
        }
        
        # Find the most impactful discriminator
        max_contributor = max(contributions, key=contributions.get)
        max_impact = contributions[max_contributor]
        
        # Log detailed analysis
        logger.info("=== INDIVIDUAL DISCRIMINATOR SCORES ===")
        for name, score in impacts.items():
            logger.info(f"{name}: {score:.4f}")
        
        logger.info("=== CONTRIBUTIONS TO FINAL SCORE ===")
        for name, contrib in contributions.items():
            logger.info(f"{name}: {contrib:.4f}")
        
        logger.info(f"=== FINAL WEIGHTED SCORE: {weighted_score:.4f} ===")
        logger.info(f"=== MOST IMPACTFUL: {max_contributor} ({max_impact:.4f}) ===")
        
        return {
            'impacts': impacts,
            'contributions': contributions,
            'weighted_score': weighted_score,
            'max_contributor': max_contributor,
            'max_impact': max_impact
        }
    
    def analyze_multiple_samples(self, texts, num_samples=10):
        """
        Analyze discriminator impact across multiple samples
        """
        logger.info(f"=== ANALYZING {num_samples} SAMPLES ===")
        
        all_analyses = []
        impact_summary = defaultdict(list)
        
        for i, text in enumerate(texts[:num_samples]):
            analysis = self.analyze_single_sample(text, i)
            all_analyses.append(analysis)
            
            # Collect metrics for summary
            for key, value in analysis['impacts'].items():
                impact_summary[key].append(value)
            for key, value in analysis['contributions'].items():
                impact_summary[key].append(value)
            impact_summary['weighted_score'].append(analysis['weighted_score'])
            impact_summary['max_contributor'].append(analysis['max_contributor'])
        
        # Calculate summary statistics
        summary_stats = {}
        for key, values in impact_summary.items():
            if key != 'max_contributor':
                summary_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Count most impactful discriminators
        contributor_counts = defaultdict(int)
        for contributor in impact_summary['max_contributor']:
            contributor_counts[contributor] += 1
        
        logger.info("=== SUMMARY STATISTICS ===")
        for key, stats in summary_stats.items():
            logger.info(f"{key}:")
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Std: {stats['std']:.4f}")
            logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        logger.info("=== MOST IMPACTFUL DISCRIMINATOR COUNTS ===")
        for contributor, count in contributor_counts.items():
            percentage = (count / num_samples) * 100
            logger.info(f"{contributor}: {count}/{num_samples} ({percentage:.1f}%)")
        
        return all_analyses, summary_stats, contributor_counts
    
    def analyze_discriminator_sensitivity(self, base_text, num_variations=5):
        """
        Analyze how sensitive each discriminator is to changes in the text
        """
        logger.info("=== DISCRIMINATOR SENSITIVITY ANALYSIS ===")
        
        variations = []
        
        # Create variations of the base text
        for i in range(num_variations):
            # Simple variation: change some values
            variation = base_text
            
            # Replace some numeric values
            import re
            def replace_with_variation(match):
                num = float(match.group())
                # Add random variation (±20%)
                variation_factor = np.random.uniform(-0.2, 0.2)
                new_num = num * (1 + variation_factor)
                return str(int(new_num) if new_num.is_integer() else round(new_num, 1))
            
            variation = re.sub(r'\d+\.?\d*', replace_with_variation, variation)
            
            # Sometimes change Good/Bad
            if random.random() < 0.3:
                variation = variation.replace("Good", "Bad") if "Good" in variation else variation.replace("Bad", "Good")
            
            variations.append(variation)
        
        # Analyze each variation
        sensitivity_results = []
        for i, variation in enumerate(variations):
            logger.info(f"Variation {i+1}: {variation[:100]}...")
            
            feedback = self.discriminators.get_multi_level_feedback(variation)
            
            result = {
                'variation_id': i,
                'token': feedback['token'],
                'sentence': feedback['sentence'],
                'row': feedback['row'],
                'feature_avg': np.mean(list(feedback['features'].values())),
                'features': feedback['features']
            }
            sensitivity_results.append(result)
        
        # Calculate sensitivity (standard deviation of scores)
        sensitivity_metrics = {}
        for discriminator in ['token', 'sentence', 'row', 'feature_avg']:
            scores = [r[discriminator] for r in sensitivity_results]
            sensitivity_metrics[discriminator] = {
                'std': np.std(scores),
                'range': np.max(scores) - np.min(scores),
                'mean': np.mean(scores)
            }
        
        logger.info("=== SENSITIVITY ANALYSIS RESULTS ===")
        for discriminator, metrics in sensitivity_metrics.items():
            logger.info(f"{discriminator}:")
            logger.info(f"  Standard Deviation: {metrics['std']:.4f}")
            logger.info(f"  Range: {metrics['range']:.4f}")
            logger.info(f"  Mean: {metrics['mean']:.4f}")
        
        return sensitivity_results, sensitivity_metrics
    
    def visualize_impact_analysis(self, analyses, save_path="discriminator_impact_analysis.png"):
        """
        Create visualizations of discriminator impact
        """
        logger.info("=== CREATING VISUALIZATIONS ===")
        
        # Prepare data for plotting
        impact_data = []
        contribution_data = []
        
        for i, analysis in enumerate(analyses):
            for discriminator, score in analysis['impacts'].items():
                impact_data.append({
                    'sample_id': i,
                    'discriminator': discriminator,
                    'score': score
                })
            
            for contributor, contrib in analysis['contributions'].items():
                contribution_data.append({
                    'sample_id': i,
                    'contributor': contributor,
                    'contribution': contrib
                })
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Individual discriminator scores
        impact_df = pd.DataFrame(impact_data)
        sns.boxplot(data=impact_df, x='discriminator', y='score', ax=axes[0,0])
        axes[0,0].set_title('Discriminator Score Distribution')
        axes[0,0].set_ylabel('Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Contributions to final score
        contrib_df = pd.DataFrame(contribution_data)
        sns.boxplot(data=contrib_df, x='contributor', y='contribution', ax=axes[0,1])
        axes[0,1].set_title('Contribution to Final Score')
        axes[0,1].set_ylabel('Contribution')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Score trends across samples
        for discriminator in impact_df['discriminator'].unique():
            data = impact_df[impact_df['discriminator'] == discriminator]
            axes[1,0].plot(data['sample_id'], data['score'], label=discriminator, marker='o')
        axes[1,0].set_title('Score Trends Across Samples')
        axes[1,0].set_xlabel('Sample ID')
        axes[1,0].set_ylabel('Score')
        axes[1,0].legend()
        
        # Plot 4: Final weighted scores
        weighted_scores = [a['weighted_score'] for a in analyses]
        axes[1,1].plot(range(len(weighted_scores)), weighted_scores, marker='o', color='red')
        axes[1,1].set_title('Final Weighted Quality Scores')
        axes[1,1].set_xlabel('Sample ID')
        axes[1,1].set_ylabel('Weighted Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
        
        return fig

def load_and_analyze_heloc_data(csv_file="heloc.csv", num_samples=20):
    """
    Load HELOC data and analyze discriminator impact
    """
    logger.info("=== LOADING HELOC DATA FOR ANALYSIS ===")
    
    # Load data
    df = pd.read_csv(csv_file)
    df = df.head(num_samples)
    
    # Format as text
    real_texts = df.apply(format_row, axis=1).tolist()
    
    return real_texts

def main():
    """
    Main function to demonstrate discriminator impact analysis
    """
    logger.info("=== DISCRIMINATOR IMPACT ANALYSIS ===")
    
    try:
        # Load trained discriminators
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Check if discriminators exist
        discriminator_path = "./hierarchical_discriminators_heloc"
        if os.path.exists(f"{discriminator_path}.pth"):
            logger.info("Loading trained HELOC discriminators...")
            discriminators = HierarchicalDiscriminatorSystem(device=device, dataset_type="heloc")
            discriminators.load_discriminators(discriminator_path)
        else:
            logger.warning("No trained discriminators found. Using untrained ones for demonstration.")
            discriminators = HierarchicalDiscriminatorSystem(device=device, dataset_type="heloc")
        
        # Initialize analyzer
        analyzer = DiscriminatorImpactAnalyzer(discriminators, device)
        
        # Load HELOC data
        real_texts = load_and_analyze_heloc_data(num_samples=10)
        
        # Analyze multiple samples
        analyses, summary_stats, contributor_counts = analyzer.analyze_multiple_samples(real_texts, num_samples=10)
        
        # Analyze sensitivity
        base_text = real_texts[0]
        sensitivity_results, sensitivity_metrics = analyzer.analyze_discriminator_sensitivity(base_text)
        
        # Create visualizations
        analyzer.visualize_impact_analysis(analyses)
        
        # Print final insights
        logger.info("=== KEY INSIGHTS ===")
        
        # Most impactful discriminator
        most_common = max(contributor_counts, key=contributor_counts.get)
        logger.info(f"Most impactful discriminator: {most_common}")
        
        # Most sensitive discriminator
        most_sensitive = max(sensitivity_metrics, key=lambda x: sensitivity_metrics[x]['std'])
        logger.info(f"Most sensitive discriminator: {most_sensitive}")
        
        # Average quality scores
        avg_quality = np.mean([a['weighted_score'] for a in analyses])
        logger.info(f"Average quality score: {avg_quality:.4f}")
        
        logger.info("=== ANALYSIS COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 