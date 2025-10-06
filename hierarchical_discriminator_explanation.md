# Hierarchical Discriminator System: Complete Technical Explanation

## Overview

The hierarchical discriminator system evaluates synthetic data at **four distinct levels**, each focusing on different aspects of data quality. This multi-level approach provides fine-grained feedback that helps the generator produce more realistic and coherent synthetic data.

## 1. Token-Level Discriminator

### **What it does:**
Evaluates individual **tokens** (words/subwords) for plausibility and coherence.

### **Architecture:**
```python
class TokenLevelDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Convert tokens to vectors
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)  # Process sequence
        self.classifier = nn.Sequential(  # Final classification
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )
```

### **How it works:**
1. **Tokenization**: Converts text to token IDs (e.g., "age is 25" → [1234, 567, 890])
2. **Embedding**: Maps each token to a 768-dimensional vector
3. **LSTM Processing**: Bidirectional LSTM processes the sequence to understand context
4. **Pooling**: Averages all token representations
5. **Classification**: Final neural network outputs a score between 0 and 1

### **Loss Function:**
```python
loss = F.binary_cross_entropy(prediction, label)
```

**Mathematical Form:**
$$\mathcal{L}_{\text{token}} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

Where:
- $y_i$ = 1 for real tokens, 0 for synthetic tokens
- $\hat{y}_i$ = predicted probability (0-1)

### **What it learns to distinguish:**

**Good Results (High Score):**
- Realistic token sequences: "age is 25", "bmi is 28.5"
- Proper grammar: "is" after feature names
- Valid numbers: "25", "28.5", "0", "1"

**Bad Results (Low Score):**
- Nonsensical tokens: "xyz is abc"
- Invalid numbers: "age is 999999"
- Grammar errors: "age 25", "bmi is"
- Out-of-vocabulary tokens

### **Example:**
```
Input: "age is 25, bmi is 28.5"
Tokens: ["age", "is", "25", ",", "bmi", "is", "28.5"]
Score: 0.85 (high - realistic tokens)

Input: "age is abc, bmi is xyz"
Tokens: ["age", "is", "abc", ",", "bmi", "is", "xyz"]
Score: 0.12 (low - unrealistic tokens)
```

---

## 2. Sentence-Level Discriminator

### **What it does:**
Evaluates **feature-value pairs** (individual sentences) for realism and consistency.

### **Architecture:**
```python
class SentenceLevelDiscriminator(nn.Module):
    def __init__(self, model_name="gpt2", hidden_dim=256):
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, output_hidden_states=True
        )  # Pre-trained language model
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
```

### **How it works:**
1. **Sentence Extraction**: Splits text into individual feature-value pairs
2. **BERT Encoding**: Uses pre-trained BERT to understand sentence meaning
3. **CLS Token**: Uses the [CLS] token representation (768-dimensional)
4. **Classification**: Final neural network outputs a score

### **Loss Function:**
Same binary cross-entropy as token-level.

### **What it learns to distinguish:**

**Good Results (High Score):**
- Realistic feature-value pairs: "age is 45", "bmi is 27.3"
- Valid ranges: "age is 25" (reasonable age)
- Proper formatting: "feature is value"

**Bad Results (Low Score):**
- Impossible values: "age is 150", "bmi is 500"
- Invalid combinations: "age is male", "gender is 25"
- Format errors: "age 25", "bmi is"

### **Example:**
```
Input: "age is 45"
Score: 0.92 (high - realistic age)

Input: "age is 150"
Score: 0.08 (low - unrealistic age)

Input: "age is male"
Score: 0.15 (low - wrong data type)
```

---

## 3. Row-Level Discriminator

### **What it does:**
Evaluates **complete data rows** for overall coherence and logical consistency.

### **Architecture:**
```python
class RowLevelDiscriminator(nn.Module):
    def __init__(self, model_name="gpt2", hidden_dim=256):
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, output_hidden_states=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
```

### **How it works:**
1. **Full Text Processing**: Processes the entire row as one text
2. **BERT Understanding**: Captures relationships between all features
3. **Global Coherence**: Evaluates if the entire row makes sense together
4. **Classification**: Outputs overall quality score

### **Loss Function:**
Same binary cross-entropy.

### **What it learns to distinguish:**

**Good Results (High Score):**
- Coherent combinations: Young person with low BMI, healthy person with normal values
- Logical consistency: All features align with a realistic profile
- Complete information: All required features present

**Bad Results (Low Score):**
- Contradictory combinations: Young person with heart disease, low BMI with diabetes
- Missing information: Incomplete feature sets
- Inconsistent patterns: Values that don't make sense together

### **Example:**
```
Input: "age is 25, bmi is 22.5, hypertension is 0, heart_disease is 0"
Score: 0.88 (high - healthy young person profile)

Input: "age is 25, bmi is 22.5, hypertension is 1, heart_disease is 1"
Score: 0.23 (low - unlikely for healthy young person)

Input: "age is 25, bmi is 45.2, hypertension is 0"
Score: 0.31 (low - missing features, high BMI for age)
```

---

## 4. Feature-Level Discriminators

### **What it does:**
Evaluates **correlations between related features** using domain knowledge.

### **Architecture:**
```python
class FeatureWiseDiscriminator(nn.Module):
    def __init__(self, feature_name, input_dim=768, hidden_dim=256):
        self.feature_name = feature_name
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
```

### **Domain-Specific Features:**

**Diabetes Dataset:**
- `age_bmi`: Age and Body Mass Index correlation
- `glucose_hba1c`: Blood glucose and HbA1c level correlation
- `hypertension_heart`: Hypertension and heart disease correlation
- `smoking_diabetes`: Smoking history and diabetes correlation

**HELOC Dataset:**
- `risk_estimate`: Risk estimation features correlation
- `trade_history`: Trading history features correlation
- `delinquency`: Delinquency-related features correlation
- `credit_utilization`: Credit utilization features correlation

### **How it works:**
1. **Feature Extraction**: Extracts embeddings for specific feature combinations
2. **Correlation Analysis**: Evaluates if related features have realistic relationships
3. **Domain Knowledge**: Applies medical/financial domain expertise
4. **Classification**: Outputs correlation plausibility score

### **Loss Function:**
Same binary cross-entropy.

### **What it learns to distinguish:**

**Good Results (High Score):**
- Realistic correlations: High age with higher BMI, high glucose with high HbA1c
- Domain-appropriate relationships: Smoking with diabetes risk
- Logical feature combinations: Hypertension with heart disease

**Bad Results (Low Score):**
- Impossible correlations: Young age with very high BMI
- Contradictory relationships: Low glucose with high HbA1c
- Domain violations: Unrealistic medical combinations

### **Example (Diabetes):**
```
Input: "age is 65, bmi is 32.1, blood_glucose_level is 180, HbA1c_level is 7.2"
age_bmi Score: 0.85 (high - realistic for older person)
glucose_hba1c Score: 0.92 (high - correlated values)

Input: "age is 25, bmi is 45.2, blood_glucose_level is 80, HbA1c_level is 8.5"
age_bmi Score: 0.12 (low - unrealistic BMI for age)
glucose_hba1c Score: 0.08 (low - contradictory values)
```

---

## Training Process

### **Data Preparation:**
```python
# Real data gets label 1, synthetic data gets label 0
real_labels = torch.ones(len(real_texts))
synthetic_labels = torch.zeros(len(synthetic_texts))
all_labels = torch.cat([real_labels, synthetic_labels])
```

### **Training Loop:**
```python
for epoch in range(epochs):
    for i, text in enumerate(all_texts):
        # Get prediction from discriminator
        prediction = discriminator(text)
        
        # Calculate loss
        loss = F.binary_cross_entropy(prediction, label)
        
        # Update discriminator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### **What Each Discriminator Learns:**

1. **Token-Level**: Learns vocabulary patterns and grammar rules
2. **Sentence-Level**: Learns realistic feature-value ranges and formats
3. **Row-Level**: Learns logical combinations and completeness
4. **Feature-Level**: Learns domain-specific correlations and relationships

---

## Multi-Level Feedback Integration

### **Feedback Aggregation:**
```python
feedback = {
    "token": token_score,           # 0-1 score
    "sentence": sentence_score,     # 0-1 score
    "row": row_score,              # 0-1 score
    "features": {                  # Dictionary of feature scores
        "age_bmi": score,
        "glucose_hba1c": score,
        # ... etc
    }
}
```

### **Weighted Combination:**
```python
composite_score = (
    0.2 * token_score +
    0.3 * sentence_score +
    0.3 * row_score +
    0.2 * feature_score
)
```

### **Loss Integration:**
```python
# Convert feedback to loss (lower feedback = higher loss)
feedback_loss = 1.0 - composite_score
total_loss = language_model_loss + 0.1 * feedback_loss
```

---

## Key Insights

### **Why Multi-Level is Important:**
1. **Token-Level**: Catches basic errors (wrong words, grammar)
2. **Sentence-Level**: Ensures realistic individual values
3. **Row-Level**: Maintains overall coherence and completeness
4. **Feature-Level**: Preserves domain knowledge and correlations

### **Training Strategy:**
- **Binary Classification**: Each discriminator learns to distinguish real vs synthetic
- **Sigmoid Output**: All scores between 0 and 1 for easy interpretation
- **Weighted Combination**: Different levels have different importance
- **Domain Knowledge**: Feature discriminators capture expert knowledge

### **Quality Assessment:**
- **High Scores (0.7-1.0)**: Excellent quality, realistic data
- **Medium Scores (0.4-0.7)**: Acceptable quality, minor issues
- **Low Scores (0.0-0.4)**: Poor quality, needs improvement

This hierarchical approach ensures that synthetic data is evaluated comprehensively, from individual tokens to complete records, leading to much higher quality synthetic datasets.
