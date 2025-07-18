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
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import logging
import sys
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hierarchical_vs_normal_comparison.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveComparison:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.hierarchical_model = None
        self.normal_model = None
        self.hierarchical_tokenizer = None
        self.normal_tokenizer = None
        
        # Data
        self.real_data = None
        self.hierarchical_synthetic = None
        self.normal_synthetic = None
        
    def load_models(self):
        """Load both hierarchical and normal GPT-2 models"""
        logger.info("Loading hierarchical model...")
        try:
            self.hierarchical_tokenizer = AutoTokenizer.from_pretrained('./gpt2_finetuned_diabetes_hierarchical')
            self.hierarchical_model = AutoModelForCausalLM.from_pretrained('./gpt2_finetuned_diabetes_hierarchical').to(self.device)
            if not self.hierarchical_tokenizer.pad_token:
                self.hierarchical_tokenizer.pad_token = self.hierarchical_tokenizer.eos_token
            logger.info("✓ Hierarchical model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load hierarchical model: {e}")
            
        logger.info("Loading normal model...")
        try:
            self.normal_tokenizer = AutoTokenizer.from_pretrained('./gpt2_finetuned_diabetes')
            self.normal_model = AutoModelForCausalLM.from_pretrained('./gpt2_finetuned_diabetes').to(self.device)
            if not self.normal_tokenizer.pad_token:
                self.normal_tokenizer.pad_token = self.normal_tokenizer.eos_token
            logger.info("✓ Normal model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load normal model: {e}")
    
    def load_data(self):
        """Load real and synthetic datasets"""
        logger.info("Loading datasets...")
        
        # Load real data
        try:
            self.real_data = pd.read_csv('diabetes.csv')
            logger.info(f"✓ Real data loaded: {len(self.real_data)} samples")
        except Exception as e:
            logger.error(f"✗ Failed to load real data: {e}")
            
        # Load hierarchical synthetic data
        try:
            self.hierarchical_synthetic = pd.read_csv('output_hierarchical_clean.csv')
            logger.info(f"✓ Hierarchical synthetic data loaded: {len(self.hierarchical_synthetic)} samples")
        except Exception as e:
            logger.error(f"✗ Failed to load hierarchical synthetic data: {e}")
            
        # Load normal synthetic data
        try:
            self.normal_synthetic = pd.read_csv('output_diabetes_clean.csv')
            logger.info(f"✓ Normal synthetic data loaded: {len(self.normal_synthetic)} samples")
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
    
    def evaluate_model_performance(self, X_train, y_train, X_test, y_test, model_name="Model"):
        """Evaluate model performance with comprehensive metrics"""
        logger.info(f"Evaluating {model_name}...")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Initialize models with class weights
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight=weight_dict
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight=weight_dict,
                max_iter=1000
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"  Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
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
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': class_report,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            logger.info(f"    {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def cross_validate_performance(self, X, y, model_name="Model"):
        """Perform cross-validation for robust evaluation"""
        logger.info(f"Cross-validating {model_name}...")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y), 
            y=y
        )
        weight_dict = dict(zip(np.unique(y), class_weights))
        
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight=weight_dict
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight=weight_dict,
                max_iter=1000
            )
        }
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            logger.info(f"  Cross-validating {model_name}...")
            
            # Cross-validate
            cv_scores = cross_val_score(
                model, X, y, 
                cv=skf, 
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            cv_results[model_name] = {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            logger.info(f"    {model_name} - Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_results
    
    def compare_approaches(self):
        """Main comparison function"""
        logger.info("=== COMPREHENSIVE HIERARCHICAL vs NORMAL COMPARISON ===")
        
        # Load models and data
        self.load_models()
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
        
        # Baseline: Train on real, test on real
        logger.info("\n=== BASELINE: REAL DATA PERFORMANCE ===")
        baseline_results = self.evaluate_model_performance(
            X_train_real, y_train_real, X_test_real, y_test_real, "Real Data Baseline"
        )
        
        # Cross-validate baseline
        baseline_cv = self.cross_validate_performance(X_real, y_real, "Real Data Baseline")
        
        # Compare with hierarchical synthetic data
        if self.hierarchical_synthetic is not None:
            logger.info("\n=== HIERARCHICAL SYNTHETIC DATA PERFORMANCE ===")
            X_hier, y_hier, _ = self.preprocess_data(self.hierarchical_synthetic)
            
            # Train on hierarchical synthetic, test on real
            hierarchical_results = self.evaluate_model_performance(
                X_hier, y_hier, X_test_real, y_test_real, "Hierarchical Synthetic"
            )
            
            # Cross-validate hierarchical
            hierarchical_cv = self.cross_validate_performance(X_hier, y_hier, "Hierarchical Synthetic")
            
            # Calculate utility preservation
            logger.info("\n=== HIERARCHICAL UTILITY PRESERVATION ===")
            for model_name in baseline_results.keys():
                baseline_acc = baseline_results[model_name]['accuracy']
                hierarchical_acc = hierarchical_results[model_name]['accuracy']
                utility_preservation = (hierarchical_acc / baseline_acc) * 100
                
                logger.info(f"{model_name}:")
                logger.info(f"  Baseline Accuracy: {baseline_acc:.4f}")
                logger.info(f"  Hierarchical Accuracy: {hierarchical_acc:.4f}")
                logger.info(f"  Utility Preservation: {utility_preservation:.2f}%")
        
        # Compare with normal synthetic data
        if self.normal_synthetic is not None:
            logger.info("\n=== NORMAL SYNTHETIC DATA PERFORMANCE ===")
            X_normal, y_normal, _ = self.preprocess_data(self.normal_synthetic)
            
            # Train on normal synthetic, test on real
            normal_results = self.evaluate_model_performance(
                X_normal, y_normal, X_test_real, y_test_real, "Normal Synthetic"
            )
            
            # Cross-validate normal
            normal_cv = self.cross_validate_performance(X_normal, y_normal, "Normal Synthetic")
            
            # Calculate utility preservation
            logger.info("\n=== NORMAL UTILITY PRESERVATION ===")
            for model_name in baseline_results.keys():
                baseline_acc = baseline_results[model_name]['accuracy']
                normal_acc = normal_results[model_name]['accuracy']
                utility_preservation = (normal_acc / baseline_acc) * 100
                
                logger.info(f"{model_name}:")
                logger.info(f"  Baseline Accuracy: {baseline_acc:.4f}")
                logger.info(f"  Normal Accuracy: {normal_acc:.4f}")
                logger.info(f"  Utility Preservation: {utility_preservation:.2f}%")
        
        # Direct comparison between approaches
        if self.hierarchical_synthetic is not None and self.normal_synthetic is not None:
            logger.info("\n=== DIRECT COMPARISON: HIERARCHICAL vs NORMAL ===")
            for model_name in baseline_results.keys():
                hierarchical_acc = hierarchical_results[model_name]['accuracy']
                normal_acc = normal_results[model_name]['accuracy']
                improvement = ((hierarchical_acc - normal_acc) / normal_acc) * 100
                
                logger.info(f"{model_name}:")
                logger.info(f"  Hierarchical Accuracy: {hierarchical_acc:.4f}")
                logger.info(f"  Normal Accuracy: {normal_acc:.4f}")
                logger.info(f"  Hierarchical Improvement: {improvement:+.2f}%")
        
        # Generate summary report
        self.generate_summary_report(
            baseline_results, hierarchical_results if self.hierarchical_synthetic is not None else None,
            normal_results if self.normal_synthetic is not None else None
        )
    
    def generate_summary_report(self, baseline_results, hierarchical_results, normal_results):
        """Generate a comprehensive summary report"""
        logger.info("\n=== COMPREHENSIVE SUMMARY REPORT ===")
        
        # Create summary table
        summary_data = []
        
        for model_name in baseline_results.keys():
            row = {'Model': model_name}
            
            # Baseline results
            if baseline_results and model_name in baseline_results:
                row['Baseline_Accuracy'] = baseline_results[model_name]['accuracy']
                row['Baseline_F1'] = baseline_results[model_name]['f1_score']
            
            # Hierarchical results
            if hierarchical_results and model_name in hierarchical_results:
                row['Hierarchical_Accuracy'] = hierarchical_results[model_name]['accuracy']
                row['Hierarchical_F1'] = hierarchical_results[model_name]['f1_score']
                if baseline_results and model_name in baseline_results:
                    row['Hierarchical_Utility_Preservation'] = (
                        hierarchical_results[model_name]['accuracy'] / 
                        baseline_results[model_name]['accuracy']
                    ) * 100
            
            # Normal results
            if normal_results and model_name in normal_results:
                row['Normal_Accuracy'] = normal_results[model_name]['accuracy']
                row['Normal_F1'] = normal_results[model_name]['f1_score']
                if baseline_results and model_name in baseline_results:
                    row['Normal_Utility_Preservation'] = (
                        normal_results[model_name]['accuracy'] / 
                        baseline_results[model_name]['accuracy']
                    ) * 100
            
            # Direct comparison
            if hierarchical_results and normal_results and model_name in hierarchical_results and model_name in normal_results:
                row['Hierarchical_vs_Normal_Improvement'] = (
                    (hierarchical_results[model_name]['accuracy'] - normal_results[model_name]['accuracy']) /
                    normal_results[model_name]['accuracy']
                ) * 100
            
            summary_data.append(row)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_df.to_csv('hierarchical_vs_normal_summary.csv', index=False)
        logger.info("Summary report saved to 'hierarchical_vs_normal_summary.csv'")
        
        # Print summary
        logger.info("\nSummary Table:")
        logger.info(summary_df.to_string(index=False))
        
        # Key findings
        logger.info("\n=== KEY FINDINGS ===")
        
        if hierarchical_results and normal_results:
            # Find best performing model for each approach
            best_hierarchical = max(hierarchical_results.items(), key=lambda x: x[1]['accuracy'])
            best_normal = max(normal_results.items(), key=lambda x: x[1]['accuracy'])
            
            logger.info(f"Best Hierarchical Model: {best_hierarchical[0]} ({best_hierarchical[1]['accuracy']:.4f})")
            logger.info(f"Best Normal Model: {best_normal[0]} ({best_normal[1]['accuracy']:.4f})")
            
            if best_hierarchical[1]['accuracy'] > best_normal[1]['accuracy']:
                improvement = ((best_hierarchical[1]['accuracy'] - best_normal[1]['accuracy']) / best_normal[1]['accuracy']) * 100
                logger.info(f"Hierarchical approach shows {improvement:.2f}% improvement over normal approach")
            else:
                improvement = ((best_normal[1]['accuracy'] - best_hierarchical[1]['accuracy']) / best_hierarchical[1]['accuracy']) * 100
                logger.info(f"Normal approach shows {improvement:.2f}% improvement over hierarchical approach")

if __name__ == "__main__":
    comparison = ComprehensiveComparison()
    comparison.compare_approaches() 