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
import logging
import sys
import warnings
from datetime import datetime
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balanced_comparison_simple.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleBalancedComparison:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Data
        self.real_data = None
        self.hierarchical_synthetic = None
        self.normal_synthetic = None
        
    def load_data(self):
        """Load real and synthetic datasets"""
        logger.info("Loading datasets...")
        
        # Load real data
        try:
            self.real_data = pd.read_csv('diabetes.csv')
            logger.info(f"✓ Real data loaded: {len(self.real_data)} samples")
            
            # Check class distribution
            class_dist = self.real_data['diabetes'].value_counts()
            logger.info(f"Real data class distribution:")
            logger.info(f"  Class 0 (No Diabetes): {class_dist[0]} ({class_dist[0]/len(self.real_data)*100:.1f}%)")
            logger.info(f"  Class 1 (Diabetes): {class_dist[1]} ({class_dist[1]/len(self.real_data)*100:.1f}%)")
            logger.info(f"  Imbalance ratio: {class_dist[0]/class_dist[1]:.1f}:1")
            
        except Exception as e:
            logger.error(f"✗ Failed to load real data: {e}")
            
        # Load hierarchical synthetic data
        try:
            self.hierarchical_synthetic = pd.read_csv('output_hierarchical_clean.csv')
            logger.info(f"✓ Hierarchical synthetic data loaded: {len(self.hierarchical_synthetic)} samples")
            
            class_dist = self.hierarchical_synthetic['diabetes'].value_counts()
            logger.info(f"Hierarchical synthetic class distribution:")
            logger.info(f"  Class 0: {class_dist[0]} ({class_dist[0]/len(self.hierarchical_synthetic)*100:.1f}%)")
            logger.info(f"  Class 1: {class_dist[1]} ({class_dist[1]/len(self.hierarchical_synthetic)*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"✗ Failed to load hierarchical synthetic data: {e}")
            
        # Load normal synthetic data
        try:
            self.normal_synthetic = pd.read_csv('output_diabetes_clean.csv')
            logger.info(f"✓ Normal synthetic data loaded: {len(self.normal_synthetic)} samples")
            
            class_dist = self.normal_synthetic['diabetes'].value_counts()
            logger.info(f"Normal synthetic class distribution:")
            logger.info(f"  Class 0: {class_dist[0]} ({class_dist[0]/len(self.normal_synthetic)*100:.1f}%)")
            logger.info(f"  Class 1: {class_dist[1]} ({class_dist[1]/len(self.normal_synthetic)*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"✗ Failed to load normal synthetic data: {e}")
    
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
    
    def simple_oversample(self, X, y, random_state=42):
        """Simple random oversampling of minority class"""
        logger.info("Applying simple random oversampling...")
        
        # Count samples per class
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # Get indices for each class
        majority_indices = np.where(y == majority_class)[0]
        minority_indices = np.where(y == minority_class)[0]
        
        # Calculate how many samples to add
        samples_to_add = len(majority_indices) - len(minority_indices)
        
        # Randomly sample from minority class with replacement
        np.random.seed(random_state)
        additional_indices = np.random.choice(minority_indices, size=samples_to_add, replace=True)
        
        # Combine original and additional samples
        balanced_indices = np.concatenate([np.arange(len(y)), additional_indices])
        
        X_balanced = X.iloc[balanced_indices].reset_index(drop=True)
        y_balanced = y.iloc[balanced_indices].reset_index(drop=True)
        
        logger.info(f"Oversampling applied: {len(y)} → {len(y_balanced)} samples")
        logger.info(f"New class distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def simple_undersample(self, X, y, random_state=42):
        """Simple random undersampling of majority class"""
        logger.info("Applying simple random undersampling...")
        
        # Count samples per class
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # Get indices for each class
        majority_indices = np.where(y == majority_class)[0]
        minority_indices = np.where(y == minority_class)[0]
        
        # Randomly sample from majority class
        np.random.seed(random_state)
        selected_majority_indices = np.random.choice(
            majority_indices, 
            size=len(minority_indices), 
            replace=False
        )
        
        # Combine minority and selected majority samples
        balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
        
        X_balanced = X.iloc[balanced_indices].reset_index(drop=True)
        y_balanced = y.iloc[balanced_indices].reset_index(drop=True)
        
        logger.info(f"Undersampling applied: {len(y)} → {len(y_balanced)} samples")
        logger.info(f"New class distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
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
            
        elif technique == 'oversample':
            # Simple oversampling
            X_balanced, y_balanced = self.simple_oversample(X, y)
            return X_balanced, y_balanced, None
            
        elif technique == 'undersample':
            # Simple undersampling
            X_balanced, y_balanced = self.simple_undersample(X, y)
            return X_balanced, y_balanced, None
            
        else:
            logger.warning(f"Unknown balancing technique: {technique}")
            return X, y, None
    
    def evaluate_with_balancing(self, X_train, y_train, X_test, y_test, technique='class_weight', model_name="Model"):
        """Evaluate model performance with balancing techniques"""
        logger.info(f"Evaluating {model_name} with {technique} balancing...")
        
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
            
            # Class-specific metrics
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
            logger.info(f"    Class 1 (Diabetes) - Precision: {precision_class_1:.4f}, Recall: {recall_class_1:.4f}, F1: {f1_class_1:.4f}")
        
        return results
    
    def comprehensive_balanced_evaluation(self):
        """Main evaluation function with multiple balancing techniques"""
        logger.info("=== COMPREHENSIVE BALANCED EVALUATION ===")
        
        # Load data
        self.load_data()
        
        if self.real_data is None:
            logger.error("Cannot proceed without real data")
            return
        
        # Preprocess real data
        X_real, y_real, _ = self.preprocess_data(self.real_data)
        
        # Split real data for testing
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
        )
        
        # Define balancing techniques to test
        balancing_techniques = [
            'class_weight',
            'oversample', 
            'undersample'
        ]
        
        # Store all results
        all_results = {}
        
        # 1. Baseline evaluation (no balancing)
        logger.info("\n=== BASELINE EVALUATION (NO BALANCING) ===")
        baseline_results = self.evaluate_with_balancing(
            X_train_real, y_train_real, X_test_real, y_test_real, 
            technique='class_weight', model_name="Real Data Baseline"
        )
        all_results['baseline'] = baseline_results
        
        # 2. Test each balancing technique on real data
        for technique in balancing_techniques:
            logger.info(f"\n=== REAL DATA WITH {technique.upper()} ===")
            results = self.evaluate_with_balancing(
                X_train_real, y_train_real, X_test_real, y_test_real,
                technique=technique, model_name=f"Real Data + {technique}"
            )
            all_results[f'real_{technique}'] = results
        
        # 3. Evaluate synthetic data with balancing
        if self.hierarchical_synthetic is not None:
            logger.info("\n=== HIERARCHICAL SYNTHETIC DATA EVALUATION ===")
            X_hier, y_hier, _ = self.preprocess_data(self.hierarchical_synthetic)
            
            for technique in balancing_techniques:
                logger.info(f"\n--- Hierarchical + {technique.upper()} ---")
                results = self.evaluate_with_balancing(
                    X_hier, y_hier, X_test_real, y_test_real,
                    technique=technique, model_name=f"Hierarchical + {technique}"
                )
                all_results[f'hierarchical_{technique}'] = results
        
        if self.normal_synthetic is not None:
            logger.info("\n=== NORMAL SYNTHETIC DATA EVALUATION ===")
            X_normal, y_normal, _ = self.preprocess_data(self.normal_synthetic)
            
            for technique in balancing_techniques:
                logger.info(f"\n--- Normal + {technique.upper()} ---")
                results = self.evaluate_with_balancing(
                    X_normal, y_normal, X_test_real, y_test_real,
                    technique=technique, model_name=f"Normal + {technique}"
                )
                all_results[f'normal_{technique}'] = results
        
        # Generate comprehensive summary
        self.generate_balanced_summary(all_results, balancing_techniques)
    
    def generate_balanced_summary(self, all_results, balancing_techniques):
        """Generate comprehensive summary of balanced evaluation"""
        logger.info("\n=== COMPREHENSIVE BALANCED SUMMARY ===")
        
        # Create summary DataFrame
        summary_data = []
        
        for result_key, results in all_results.items():
            for model_name, metrics in results.items():
                row = {
                    'Dataset': result_key,
                    'Model': model_name,
                    'Balancing': 'class_weight' if 'class_weight' in result_key else result_key.split('_')[-1] if '_' in result_key else 'none',
                    'Accuracy': metrics['accuracy'],
                    'Balanced_Accuracy': metrics['balanced_accuracy'],
                    'F1_Score': metrics['f1_score'],
                    'Kappa': metrics['kappa'],
                    'ROC_AUC': metrics['roc_auc'],
                    'Precision_Class_1': metrics['precision_class_1'],
                    'Recall_Class_1': metrics['recall_class_1'],
                    'F1_Class_1': metrics['f1_class_1']
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save detailed summary
        summary_df.to_csv('balanced_evaluation_summary.csv', index=False)
        logger.info("Detailed summary saved to 'balanced_evaluation_summary.csv'")
        
        # Print key findings
        logger.info("\n=== KEY FINDINGS ===")
        
        # Find best performing combinations
        best_balanced_accuracy = summary_df.loc[summary_df['Balanced_Accuracy'].idxmax()]
        best_f1_class_1 = summary_df.loc[summary_df['F1_Class_1'].idxmax()]
        best_overall_f1 = summary_df.loc[summary_df['F1_Score'].idxmax()]
        
        logger.info(f"Best Balanced Accuracy: {best_balanced_accuracy['Dataset']} + {best_balanced_accuracy['Model']} + {best_balanced_accuracy['Balancing']} ({best_balanced_accuracy['Balanced_Accuracy']:.4f})")
        logger.info(f"Best F1 for Diabetes Class: {best_f1_class_1['Dataset']} + {best_f1_class_1['Model']} + {best_f1_class_1['Balancing']} ({best_f1_class_1['F1_Class_1']:.4f})")
        logger.info(f"Best Overall F1: {best_overall_f1['Dataset']} + {best_overall_f1['Model']} + {best_overall_f1['Balancing']} ({best_overall_f1['F1_Score']:.4f})")
        
        # Compare approaches
        logger.info("\n=== APPROACH COMPARISON ===")
        
        # Compare hierarchical vs normal for each balancing technique
        for technique in balancing_techniques:
            hier_results = summary_df[(summary_df['Dataset'].str.contains('hierarchical')) & (summary_df['Balancing'] == technique)]
            normal_results = summary_df[(summary_df['Dataset'].str.contains('normal')) & (summary_df['Balancing'] == technique)]
            
            if not hier_results.empty and not normal_results.empty:
                logger.info(f"\n--- {technique.upper()} Comparison ---")
                
                for model in ['Decision Tree', 'Random Forest', 'Logistic Regression']:
                    hier_metric = hier_results[hier_results['Model'] == model]['Balanced_Accuracy'].iloc[0] if not hier_results[hier_results['Model'] == model].empty else 0
                    normal_metric = normal_results[normal_results['Model'] == model]['Balanced_Accuracy'].iloc[0] if not normal_results[normal_results['Model'] == model].empty else 0
                    
                    if hier_metric > 0 and normal_metric > 0:
                        improvement = ((hier_metric - normal_metric) / normal_metric) * 100
                        logger.info(f"{model}: Hierarchical {hier_metric:.4f} vs Normal {normal_metric:.4f} ({improvement:+.2f}%)")

if __name__ == "__main__":
    evaluation = SimpleBalancedComparison()
    evaluation.comprehensive_balanced_evaluation() 