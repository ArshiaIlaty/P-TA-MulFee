import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import logging
import sys
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('three_way_comparison_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ThreeWayComparisonEvaluation:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create results directory
        self.results_dir = 'comparison_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Data sources with clear labeling
        self.datasets = {
            'real': None,                    # Baseline: diabetes.csv
            'gpt_fine_tuned': None,          # Synthetic - GPT Fine-Tuned: output_diabetes_clean.csv
            'hierarchical_gpt': None         # Synthetic - Hierarchical GPT Fine-Tuned: output_hierarchical_clean.csv
        }
        
    def load_data(self):
        """Load all three datasets with clear labeling"""
        logger.info("=== LOADING THREE DATASETS FOR COMPARISON ===")
        
        # 1. Baseline (Real Data)
        try:
            self.datasets['real'] = pd.read_csv('diabetes.csv')
            logger.info(f"✓ Baseline (Real Data) loaded: {len(self.datasets['real'])} samples")
            
            class_dist = self.datasets['real']['diabetes'].value_counts()
            logger.info(f"  Class distribution: {class_dist[0]} ({class_dist[0]/len(self.datasets['real'])*100:.1f}%) vs {class_dist[1]} ({class_dist[1]/len(self.datasets['real'])*100:.1f}%)")
            logger.info(f"  Imbalance ratio: {class_dist[0]/class_dist[1]:.1f}:1")
            
        except Exception as e:
            logger.error(f"✗ Failed to load Baseline (Real Data): {e}")
            
        # 2. Synthetic - GPT Fine-Tuned
        try:
            self.datasets['gpt_fine_tuned'] = pd.read_csv('output_diabetes_clean.csv')
            logger.info(f"✓ Synthetic - GPT Fine-Tuned loaded: {len(self.datasets['gpt_fine_tuned'])} samples")
            
            class_dist = self.datasets['gpt_fine_tuned']['diabetes'].value_counts()
            logger.info(f"  Class distribution: {class_dist[0]} ({class_dist[0]/len(self.datasets['gpt_fine_tuned'])*100:.1f}%) vs {class_dist[1]} ({class_dist[1]/len(self.datasets['gpt_fine_tuned'])*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"✗ Failed to load Synthetic - GPT Fine-Tuned: {e}")
            
        # 3. Synthetic - Hierarchical GPT Fine-Tuned
        try:
            self.datasets['hierarchical_gpt'] = pd.read_csv('output_hierarchical_clean.csv')
            logger.info(f"✓ Synthetic - Hierarchical GPT Fine-Tuned loaded: {len(self.datasets['hierarchical_gpt'])} samples")
            
            class_dist = self.datasets['hierarchical_gpt']['diabetes'].value_counts()
            logger.info(f"  Class distribution: {class_dist[0]} ({class_dist[0]/len(self.datasets['hierarchical_gpt'])*100:.1f}%) vs {class_dist[1]} ({class_dist[1]/len(self.datasets['hierarchical_gpt'])*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"✗ Failed to load Synthetic - Hierarchical GPT Fine-Tuned: {e}")
    
    def preprocess_data(self, df, target_col='diabetes'):
        """Preprocess data for machine learning"""
        if df is None:
            return None, None, None
            
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col])
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        return X, y, label_encoders
    
    def apply_balancing_techniques(self, X, y, technique='class_weight'):
        """Apply various balancing techniques"""
        logger.info(f"Applying {technique} balancing technique...")
        
        if technique == 'class_weight':
            # Use class weights (no data modification)
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y), 
                y=y
            )
            weight_dict = dict(zip(np.unique(y), class_weights))
            logger.info(f"Class weights: {weight_dict}")
            return X, y, weight_dict
            
        elif technique == 'smote':
            # Apply SMOTE oversampling
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            logger.info(f"SMOTE applied: {len(y)} → {len(y_balanced)} samples")
            return X_balanced, y_balanced, None
            
        elif technique == 'random_oversample':
            # Random oversampling
            ros = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = ros.fit_resample(X, y)
            logger.info(f"Random oversampling applied: {len(y)} → {len(y_balanced)} samples")
            return X_balanced, y_balanced, None
            
        elif technique == 'random_undersample':
            # Random undersampling
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
            logger.info(f"Random undersampling applied: {len(y)} → {len(y_balanced)} samples")
            return X_balanced, y_balanced, None
            
        elif technique == 'smoteenn':
            # SMOTE + ENN (Edited Nearest Neighbors)
            smoteenn = SMOTEENN(random_state=42)
            X_balanced, y_balanced = smoteenn.fit_resample(X, y)
            logger.info(f"SMOTEENN applied: {len(y)} → {len(y_balanced)} samples")
            return X_balanced, y_balanced, None
            
        else:
            logger.warning(f"Unknown balancing technique: {technique}")
            return X, y, None
    
    def evaluate_with_balancing(self, X_train, y_train, X_test, y_test, technique='class_weight', dataset_name="Dataset"):
        """Evaluate model performance with balancing techniques"""
        logger.info(f"Evaluating {dataset_name} with {technique} balancing...")
        
        # Apply balancing technique
        X_train_balanced, y_train_balanced, class_weights = self.apply_balancing_techniques(
            X_train, y_train, technique
        )
        
        # Initialize models
        models = {}
        
        if technique == 'class_weight':
            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight=class_weights),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    class_weight=class_weights
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42, 
                    class_weight=class_weights,
                    max_iter=1000
                )
            }
        else:
            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"  Training {model_name}...")
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            kappa = cohen_kappa_score(y_test, y_pred)
            
            # Class-specific metrics (focus on diabetes class - minority class)
            precision_class_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            recall_class_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            f1_class_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
            
            precision_class_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall_class_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1_class_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            
            # Calculate ROC AUC if possible
            roc_auc = None
            if y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
            
            # Class-specific metrics
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'kappa': kappa,
                'roc_auc': roc_auc,
                'precision_class_0': precision_class_0,
                'recall_class_0': recall_class_0,
                'f1_class_0': f1_class_0,
                'precision_class_1': precision_class_1,
                'recall_class_1': recall_class_1,
                'f1_class_1': f1_class_1,
                'classification_report': class_report,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            logger.info(f"    {model_name} - Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_accuracy:.4f}")
            logger.info(f"    Diabetes Class - Precision: {precision_class_1:.4f}, Recall: {recall_class_1:.4f}, F1: {f1_class_1:.4f}")
        
        return results
    
    def comprehensive_three_way_evaluation(self):
        """Main evaluation function for three-way comparison"""
        logger.info("=== COMPREHENSIVE THREE-WAY COMPARISON EVALUATION ===")
        
        # Load all datasets
        self.load_data()
        
        if self.datasets['real'] is None:
            logger.error("Cannot proceed without baseline (real) data")
            return
        
        # Preprocess real data for testing
        X_real, y_real, _ = self.preprocess_data(self.datasets['real'])
        
        # Split real data for testing (we test all approaches on the same real test set)
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
        )
        
        # Define balancing techniques to test
        balancing_techniques = [
            'class_weight',
            'smote', 
            'random_oversample',
            'random_undersample',
            'smoteenn'
        ]
        
        # Store all results
        all_results = {}
        
        # 1. Baseline (Real Data) evaluation
        logger.info("\n=== BASELINE (REAL DATA) EVALUATION ===")
        for technique in balancing_techniques:
            logger.info(f"\n--- Real Data + {technique.upper()} ---")
            results = self.evaluate_with_balancing(
                X_train_real, y_train_real, X_test_real, y_test_real,
                technique=technique, dataset_name=f"Real Data + {technique}"
            )
            all_results[f'real_{technique}'] = results
        
        # 2. Synthetic - GPT Fine-Tuned evaluation
        if self.datasets['gpt_fine_tuned'] is not None:
            logger.info("\n=== SYNTHETIC - GPT FINE-TUNED EVALUATION ===")
            X_gpt, y_gpt, _ = self.preprocess_data(self.datasets['gpt_fine_tuned'])
            
            for technique in balancing_techniques:
                logger.info(f"\n--- GPT Fine-Tuned + {technique.upper()} ---")
                results = self.evaluate_with_balancing(
                    X_gpt, y_gpt, X_test_real, y_test_real,
                    technique=technique, dataset_name=f"GPT Fine-Tuned + {technique}"
                )
                all_results[f'gpt_fine_tuned_{technique}'] = results
        
        # 3. Synthetic - Hierarchical GPT Fine-Tuned evaluation
        if self.datasets['hierarchical_gpt'] is not None:
            logger.info("\n=== SYNTHETIC - HIERARCHICAL GPT FINE-TUNED EVALUATION ===")
            X_hier, y_hier, _ = self.preprocess_data(self.datasets['hierarchical_gpt'])
            
            for technique in balancing_techniques:
                logger.info(f"\n--- Hierarchical GPT + {technique.upper()} ---")
                results = self.evaluate_with_balancing(
                    X_hier, y_hier, X_test_real, y_test_real,
                    technique=technique, dataset_name=f"Hierarchical GPT + {technique}"
                )
                all_results[f'hierarchical_gpt_{technique}'] = results
        
        # Generate comprehensive summary and visualizations
        self.generate_three_way_summary(all_results, balancing_techniques)
    
    def generate_three_way_summary(self, all_results, balancing_techniques):
        """Generate comprehensive summary and visualizations for three-way comparison"""
        logger.info("\n=== COMPREHENSIVE THREE-WAY SUMMARY ===")
        
        # Create summary DataFrame
        summary_data = []
        
        for result_key, results in all_results.items():
            for model_name, metrics in results.items():
                # Extract dataset type from result_key
                if result_key.startswith('real_'):
                    dataset_type = 'Baseline (Real Data)'
                    technique = result_key.replace('real_', '')
                elif result_key.startswith('gpt_fine_tuned_'):
                    dataset_type = 'Synthetic - GPT Fine-Tuned'
                    technique = result_key.replace('gpt_fine_tuned_', '')
                elif result_key.startswith('hierarchical_gpt_'):
                    dataset_type = 'Synthetic - Hierarchical GPT Fine-Tuned'
                    technique = result_key.replace('hierarchical_gpt_', '')
                else:
                    dataset_type = 'Unknown'
                    technique = 'unknown'
                
                row = {
                    'Dataset_Type': dataset_type,
                    'Model': model_name,
                    'Balancing_Technique': technique,
                    'Accuracy': metrics['accuracy'],
                    'Balanced_Accuracy': metrics['balanced_accuracy'],
                    'F1_Score': metrics['f1_score'],
                    'Kappa': metrics['kappa'],
                    'ROC_AUC': metrics['roc_auc'],
                    'Diabetes_Precision': metrics['precision_class_1'],
                    'Diabetes_Recall': metrics['recall_class_1'],
                    'Diabetes_F1': metrics['f1_class_1']
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save detailed summary
        summary_path = os.path.join(self.results_dir, 'three_way_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Detailed summary saved to {summary_path}")
        
        # Print key findings
        logger.info("\n=== KEY FINDINGS ===")
        
        # Find best performing combinations
        best_balanced_accuracy = summary_df.loc[summary_df['Balanced_Accuracy'].idxmax()]
        best_diabetes_f1 = summary_df.loc[summary_df['Diabetes_F1'].idxmax()]
        best_overall_f1 = summary_df.loc[summary_df['F1_Score'].idxmax()]
        
        logger.info(f"Best Balanced Accuracy: {best_balanced_accuracy['Dataset_Type']} + {best_balanced_accuracy['Model']} + {best_balanced_accuracy['Balancing_Technique']} ({best_balanced_accuracy['Balanced_Accuracy']:.4f})")
        logger.info(f"Best Diabetes F1: {best_diabetes_f1['Dataset_Type']} + {best_diabetes_f1['Model']} + {best_diabetes_f1['Balancing_Technique']} ({best_diabetes_f1['Diabetes_F1']:.4f})")
        logger.info(f"Best Overall F1: {best_overall_f1['Dataset_Type']} + {best_overall_f1['Model']} + {best_overall_f1['Balancing_Technique']} ({best_overall_f1['F1_Score']:.4f})")
        
        # Generate visualizations
        self.create_visualizations(summary_df, balancing_techniques)
        
        # Compare approaches
        self.compare_approaches(summary_df, balancing_techniques)
    
    def create_visualizations(self, summary_df, balancing_techniques):
        """Create comprehensive visualizations for three-way comparison"""
        logger.info("\n=== CREATING VISUALIZATIONS ===")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Balanced Accuracy Comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Three-Way Comparison: Baseline vs GPT Fine-Tuned vs Hierarchical GPT Fine-Tuned', fontsize=16, fontweight='bold')
        
        # 1.1 Balanced Accuracy by Dataset Type
        ax1 = axes[0, 0]
        dataset_means = summary_df.groupby('Dataset_Type')['Balanced_Accuracy'].mean().sort_values(ascending=False)
        bars = ax1.bar(dataset_means.index, dataset_means.values, alpha=0.8)
        ax1.set_title('Average Balanced Accuracy by Dataset Type')
        ax1.set_ylabel('Balanced Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 1.2 Diabetes F1 Score by Dataset Type
        ax2 = axes[0, 1]
        diabetes_f1_means = summary_df.groupby('Dataset_Type')['Diabetes_F1'].mean().sort_values(ascending=False)
        bars = ax2.bar(diabetes_f1_means.index, diabetes_f1_means.values, alpha=0.8, color='orange')
        ax2.set_title('Average Diabetes F1 Score by Dataset Type')
        ax2.set_ylabel('Diabetes F1 Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 1.3 Model Performance Comparison
        ax3 = axes[1, 0]
        model_means = summary_df.groupby('Model')['Balanced_Accuracy'].mean().sort_values(ascending=False)
        bars = ax3.bar(model_means.index, model_means.values, alpha=0.8, color='green')
        ax3.set_title('Average Balanced Accuracy by Model Type')
        ax3.set_ylabel('Balanced Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 1.4 Balancing Technique Performance
        ax4 = axes[1, 1]
        technique_means = summary_df.groupby('Balancing_Technique')['Balanced_Accuracy'].mean().sort_values(ascending=False)
        bars = ax4.bar(technique_means.index, technique_means.values, alpha=0.8, color='red')
        ax4.set_title('Average Balanced Accuracy by Balancing Technique')
        ax4.set_ylabel('Balanced Accuracy')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'overview_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Detailed Comparison by Balancing Technique
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Detailed Performance Comparison by Balancing Technique', fontsize=16, fontweight='bold')
        
        for i, technique in enumerate(balancing_techniques[:6]):  # Limit to 6 techniques
            ax = axes[i//3, i%3]
            
            # Filter data for this technique
            tech_data = summary_df[summary_df['Balancing_Technique'] == technique]
            
            if not tech_data.empty:
                # Create grouped bar plot
                x = np.arange(len(tech_data['Model'].unique()))
                width = 0.25
                
                datasets = ['Baseline (Real Data)', 'Synthetic - GPT Fine-Tuned', 'Synthetic - Hierarchical GPT Fine-Tuned']
                colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']
                
                for j, dataset in enumerate(datasets):
                    dataset_data = tech_data[tech_data['Dataset_Type'] == dataset]
                    if not dataset_data.empty:
                        balanced_acc = dataset_data.groupby('Model')['Balanced_Accuracy'].mean().values
                        ax.bar(x + j*width, balanced_acc, width, label=dataset, color=colors[j], alpha=0.8)
                
                ax.set_xlabel('Model Type')
                ax.set_ylabel('Balanced Accuracy')
                ax.set_title(f'{technique.upper()} Balancing')
                ax.set_xticks(x + width)
                ax.set_xticklabels(tech_data['Model'].unique(), rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'detailed_technique_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Summary Table
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare summary table data
        summary_table = summary_df.groupby(['Dataset_Type', 'Model']).agg({
            'Balanced_Accuracy': ['mean', 'std'],
            'Diabetes_F1': ['mean', 'std'],
            'ROC_AUC': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns]
        summary_table = summary_table.reset_index()
        
        # Create table
        table_data = []
        for _, row in summary_table.iterrows():
            table_data.append([
                row['Dataset_Type'],
                row['Model'],
                f"{row['Balanced_Accuracy_mean']:.4f} ± {row['Balanced_Accuracy_std']:.4f}",
                f"{row['Diabetes_F1_mean']:.4f} ± {row['Diabetes_F1_std']:.4f}",
                f"{row['ROC_AUC_mean']:.4f} ± {row['ROC_AUC_std']:.4f}"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Dataset Type', 'Model', 'Balanced Accuracy', 'Diabetes F1', 'ROC AUC'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('Summary Statistics by Dataset Type and Model', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(os.path.join(self.results_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualizations saved to {self.results_dir}/")
    
    def compare_approaches(self, summary_df, balancing_techniques):
        """Compare the three approaches systematically"""
        logger.info("\n=== SYSTEMATIC APPROACH COMPARISON ===")
        
        # Compare GPT Fine-Tuned vs Hierarchical GPT Fine-Tuned
        logger.info("\n--- GPT Fine-Tuned vs Hierarchical GPT Fine-Tuned ---")
        
        for technique in balancing_techniques:
            gpt_data = summary_df[(summary_df['Dataset_Type'] == 'Synthetic - GPT Fine-Tuned') & 
                                 (summary_df['Balancing_Technique'] == technique)]
            hier_data = summary_df[(summary_df['Dataset_Type'] == 'Synthetic - Hierarchical GPT Fine-Tuned') & 
                                  (summary_df['Balancing_Technique'] == technique)]
            
            if not gpt_data.empty and not hier_data.empty:
                logger.info(f"\n{technique.upper()} Comparison:")
                
                for model in ['Decision Tree', 'Random Forest', 'Logistic Regression']:
                    gpt_metric = gpt_data[gpt_data['Model'] == model]['Balanced_Accuracy'].iloc[0] if not gpt_data[gpt_data['Model'] == model].empty else 0
                    hier_metric = hier_data[hier_data['Model'] == model]['Balanced_Accuracy'].iloc[0] if not hier_data[hier_data['Model'] == model].empty else 0
                    
                    if gpt_metric > 0 and hier_metric > 0:
                        improvement = ((hier_metric - gpt_metric) / gpt_metric) * 100
                        logger.info(f"  {model}: Hierarchical {hier_metric:.4f} vs GPT {gpt_metric:.4f} ({improvement:+.2f}%)")
        
        # Compare against baseline
        logger.info("\n--- Comparison Against Baseline (Real Data) ---")
        
        baseline_means = summary_df[summary_df['Dataset_Type'] == 'Baseline (Real Data)'].groupby('Model')['Balanced_Accuracy'].mean()
        gpt_means = summary_df[summary_df['Dataset_Type'] == 'Synthetic - GPT Fine-Tuned'].groupby('Model')['Balanced_Accuracy'].mean()
        hier_means = summary_df[summary_df['Dataset_Type'] == 'Synthetic - Hierarchical GPT Fine-Tuned'].groupby('Model')['Balanced_Accuracy'].mean()
        
        for model in baseline_means.index:
            baseline_acc = baseline_means[model]
            gpt_acc = gpt_means.get(model, 0)
            hier_acc = hier_means.get(model, 0)
            
            gpt_preservation = (gpt_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
            hier_preservation = (hier_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
            
            logger.info(f"{model}:")
            logger.info(f"  Baseline: {baseline_acc:.4f}")
            logger.info(f"  GPT Fine-Tuned: {gpt_acc:.4f} ({gpt_preservation:.1f}% preservation)")
            logger.info(f"  Hierarchical GPT: {hier_acc:.4f} ({hier_preservation:.1f}% preservation)")

if __name__ == "__main__":
    evaluation = ThreeWayComparisonEvaluation()
    evaluation.comprehensive_three_way_evaluation() 