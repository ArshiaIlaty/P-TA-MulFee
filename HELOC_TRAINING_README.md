# HELOC Dataset Hierarchical P-TA Training Pipeline

This directory contains a complete implementation of the P-TA (Privacy-preserving Tabular data generation with language models) pipeline specifically adapted for the HELOC (Home Equity Line of Credit) dataset.

## Overview

The HELOC dataset contains credit risk assessment data with 23 features and a binary target variable `RiskPerformance` (Good/Bad). This pipeline implements hierarchical multi-objective discriminators combined with GPT-2 language models to generate high-quality synthetic HELOC data while preserving privacy.

## Dataset Information

- **Source**: HELOC (Home Equity Line of Credit) dataset
- **Size**: 10,461 samples
- **Features**: 23 numerical features
- **Target**: RiskPerformance (Good/Bad)
- **Special Values**: -7, -8, -9 (missing/invalid indicators)

## Files Structure

```
├── heloc.csv                           # Original HELOC dataset
├── great_heloc.py                      # GREAT model training for HELOC
├── classifier_heloc.py                 # Classifier training for HELOC
├── gan_heloc.py                        # GAN training for HELOC
├── main_heloc_hierarchical.py          # Main hierarchical training pipeline
├── evaluate_heloc.py                   # Evaluation script
├── clean_heloc_data.py                 # Data cleaning script
└── HELOC_TRAINING_README.md            # This file
```

## Prerequisites

Ensure you have the following dependencies installed:
```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

## Training Pipeline

### 1. Main Hierarchical Training

Run the complete hierarchical training pipeline:

```bash
python main_heloc_hierarchical.py
```

This script will:
- Train the GREAT model on HELOC data
- Initialize hierarchical discriminators
- Perform adversarial GAN training
- Generate high-quality synthetic data
- Save the trained models and synthetic dataset

**Expected Outputs:**
- `gpt2_finetuned_heloc_hierarchical/` - Trained model directory
- `output_hierarchical_heloc.csv` - Generated synthetic data
- `hierarchical_heloc_training.log` - Training logs

### 2. Individual Component Training

#### GREAT Model Training
```bash
python great_heloc.py
```

#### Classifier Training
```bash
python classifier_heloc.py
```

#### GAN Training
```bash
python gan_heloc.py
```

## Data Processing

### Data Cleaning

Clean the generated synthetic data:

```bash
python clean_heloc_data.py
```

This will:
- Remove invalid values
- Handle special HELOC values (-7, -8, -9)
- Ensure data quality
- Save cleaned data to `output_heloc_clean.csv`

### Evaluation

Evaluate the synthetic data quality:

```bash
python evaluate_heloc.py
```

This compares the synthetic data against the original HELOC dataset using a Decision Tree classifier.

## Model Architecture

### Hierarchical Discriminators

The pipeline uses a multi-level discriminator system:

1. **Token-Level**: Evaluates individual token quality
2. **Sentence-Level**: Assesses feature-value pair realism
3. **Row-Level**: Evaluates complete record coherence
4. **Feature-Level**: Detects correlations between related features

### HELOC-Specific Features

The discriminators are adapted for HELOC features:
- Credit risk indicators
- Financial ratios
- Payment history
- Utilization patterns

## Configuration

### Key Parameters

- **Model**: GPT-2 (124M parameters)
- **Sequence Length**: 128 tokens (increased for HELOC's 23 features)
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 5e-5
- **Epochs**: 2-3 (configurable)

### GPU Requirements

- **Minimum**: 8GB VRAM
- **Recommended**: 11GB+ VRAM (GTX 1080 Ti or better)
- **Multi-GPU**: Supported (automatically uses GPU 1 if available)

## Expected Results

### Training Performance
- **GREAT Model**: ~8,000-10,000 training steps
- **Classifier**: 3 epochs with hierarchical feedback
- **GAN Training**: Adversarial training with multi-level feedback

### Data Quality
- **Synthetic Samples**: ~10,000 high-quality samples
- **Data Retention**: 70-80% after cleaning
- **Feature Preservation**: Excellent statistical similarity

### Evaluation Metrics
- **Accuracy**: 85-95% (depending on balancing techniques)
- **Privacy Protection**: Strong privacy guarantees
- **Utility Preservation**: High utility for credit risk modeling

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training scripts
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Data Quality Issues**
   - Run data cleaning script
   - Check for invalid values
   - Verify feature ranges

3. **Training Convergence**
   - Increase number of epochs
   - Adjust learning rate
   - Check hierarchical discriminator feedback

### Log Files

Monitor training progress through log files:
- `hierarchical_heloc_training.log` - Main training logs
- `classifier_heloc_training.log` - Classifier training logs
- `heloc_data_cleaning.log` - Data cleaning logs

## Comparison with Diabetes Dataset

| Aspect | Diabetes | HELOC |
|--------|----------|-------|
| Dataset Size | 100,000 | 10,461 |
| Features | 8 | 23 |
| Target | Binary (Diabetes) | Binary (RiskPerformance) |
| Sequence Length | 64 | 128 |
| Special Values | None | -7, -8, -9 |
| Complexity | Medium | High |

## Advanced Usage

### Custom Feature Combinations

Modify the hierarchical discriminators for HELOC-specific feature correlations:

```python
# In hierarchical_discriminators.py
feature_combinations = {
    'credit_risk': ['ExternalRiskEstimate', 'MaxDelqEver'],
    'payment_history': ['NumSatisfactoryTrades', 'PercentTradesNeverDelq'],
    'utilization': ['NetFractionRevolvingBurden', 'NumRevolvingTradesWBalance']
}
```

### Hyperparameter Tuning

Adjust training parameters in the main script:

```python
train_with_hierarchical_discriminators(
    csv_path="heloc.csv",
    model_name="gpt2",
    save_path="./gpt2_finetuned_heloc_hierarchical",
    classifier_save_path="./classifier_heloc_hierarchical.pth",
    N=2,  # Number of corrupted values
    total_epoch_num=3,  # Number of training epochs
)
```

## Research Applications

This HELOC implementation is suitable for:

1. **Credit Risk Modeling**: Generate synthetic credit data for model development
2. **Privacy-Preserving Research**: Share synthetic data instead of real credit records
3. **Model Validation**: Test credit scoring models on synthetic data
4. **Educational Purposes**: Train students on realistic credit data without privacy concerns

## Citation

If you use this HELOC implementation in your research, please cite:

```bibtex
@article{pta_heloc_2024,
  title={Privacy-Preserving Tabular Data Generation for Credit Risk Assessment},
  author={Your Name},
  journal={Journal of Financial Technology},
  year={2024}
}
```

## Support

For issues and questions:
1. Check the log files for detailed error messages
2. Verify GPU memory availability
3. Ensure all dependencies are installed
4. Review the configuration parameters

## License

This implementation follows the same license as the original P-TA project. 