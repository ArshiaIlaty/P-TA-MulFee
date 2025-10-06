# Understanding Data Concepts: Tokens, Sentences, Rows, and Features

## 1. **Tokens** (Smallest Units of Data)

**Definition**: Tokens are the smallest meaningful units of data that can be processed individually.

### Examples from HELOC Dataset:

**Text Tokens:**
- From column name: `["RiskPerformance", "ExternalRiskEstimate", "MSinceOldestTradeOpen"]`
- From data value: `["Bad", "55", "144"]`

**Numeric Tokens:**
- Individual numbers: `55`, `144`, `4`, `84`
- Special values: `-7`, `-8`, `-9` (missing data indicators)

**String Tokens:**
- Categories: `"Good"`, `"Bad"`
- Column names: `"NumSatisfactoryTrades"`, `"PercentTradesNeverDelq"`

---

## 2. **Sentences vs Rows** - The Key Difference

**IMPORTANT**: In data science, "sentences" and "rows" often refer to the **SAME THING** - they're just different perspectives of the same data!

### **Rows** (Physical Structure)
- **Definition**: A horizontal line in a spreadsheet or CSV file
- **Perspective**: How data is **stored and displayed**
- **Example**: Line 2 in your CSV file

### **Sentences** (Logical Structure)  
- **Definition**: A complete data record that tells a story
- **Perspective**: How data is **interpreted and processed**
- **Example**: One complete loan applicant record

### **Visual Example:**

```
Row 1 (Header):    RiskPerformance,ExternalRiskEstimate,MSinceOldestTradeOpen,...
Row 2 (Data):      Bad,55,144,4,84,20,3,0,83,2,3,5,23,1,43,0,0,0,33,-8,8,1,1,69
Row 3 (Data):      Bad,61,58,15,41,2,4,4,100,-7,0,8,7,0,67,0,0,0,0,-8,0,-8,-8,0
Row 4 (Data):      Good,54,88,7,37,25,0,0,92,9,4,6,26,3,58,0,4,4,89,76,7,7,2,100
```

**Same Data, Different Names:**
- **Row 2** = **Sentence 1** = **Record 1** = **Observation 1**
- **Row 3** = **Sentence 2** = **Record 2** = **Observation 2**
- **Row 4** = **Sentence 3** = **Record 3** = **Observation 3**

### **Why Both Terms Exist:**

1. **"Row"** - Used when talking about:
   - File structure
   - Data storage
   - Spreadsheet operations
   - CSV processing

2. **"Sentence"** - Used when talking about:
   - Machine learning training
   - Data processing
   - Natural language processing
   - When treating data as "stories" or "examples"

---

## 3. **Complete Data Records** (Sentences/Rows)

**Definition**: A complete record or observation - one full row of data that tells a complete story about a single entity.

### Example from HELOC Dataset:

**One Complete Record (Row/Sentence):**
```
Bad,55,144,4,84,20,3,0,83,2,3,5,23,1,43,0,0,0,33,-8,8,1,1,69
```

This record tells the complete story of one loan applicant:
- **RiskPerformance**: Bad (loan defaulted)
- **ExternalRiskEstimate**: 55 (risk score)
- **MSinceOldestTradeOpen**: 144 months
- **MSinceMostRecentTradeOpen**: 4 months
- **AverageMInFile**: 84 months
- **NumSatisfactoryTrades**: 20 trades
- ... and so on for all 24 features

**Another Record:**
```
Good,54,88,7,37,25,0,0,92,9,4,6,26,3,58,0,4,4,89,76,7,7,2,100
```

This tells the story of a successful loan applicant.

---

## 4. **Features** (Columns/Variables)

**Definition**: Features are the characteristics or attributes that describe each observation. They are the columns in a dataset.

### Examples from HELOC Dataset:

**Target Feature (What we want to predict):**
- `RiskPerformance`: "Good" or "Bad" (loan performance)

**Predictive Features (What we use to make predictions):**

1. **Risk Assessment Features:**
   - `ExternalRiskEstimate`: External risk score (55, 61, 67, etc.)
   - `MaxDelqEver`: Maximum delinquency ever (5, 8, 6, etc.)

2. **Credit History Features:**
   - `MSinceOldestTradeOpen`: Months since oldest trade opened (144, 58, 66, etc.)
   - `NumTotalTrades`: Total number of trades (23, 7, 9, etc.)
   - `NumSatisfactoryTrades`: Number of satisfactory trades (20, 2, 9, etc.)

3. **Recent Activity Features:**
   - `MSinceMostRecentTradeOpen`: Months since most recent trade (4, 15, 5, etc.)
   - `NumInqLast6M`: Number of inquiries in last 6 months (0, 0, 4, etc.)

4. **Trade Type Features:**
   - `PercentInstallTrades`: Percentage of installment trades (43, 67, 44, etc.)
   - `NumRevolvingTradesWBalance`: Number of revolving trades with balance (8, 0, 4, etc.)

5. **Burden Features:**
   - `NetFractionRevolvingBurden`: Net fraction revolving burden (33, 0, 53, etc.)
   - `NetFractionInstallBurden`: Net fraction installment burden (-8, -8, 66, etc.)

---

## Summary Table

| Concept | Definition | Example from HELOC |
|---------|------------|-------------------|
| **Token** | Smallest unit of data | `"55"`, `"Bad"`, `"ExternalRiskEstimate"` |
| **Row/Sentence** | Complete data record | One full row describing a loan applicant |
| **Feature** | Column/variable | `RiskPerformance`, `ExternalRiskEstimate`, etc. |

---

## **Key Takeaway:**

**Row = Sentence = Record = Observation**

They all mean the same thing! It's just different terminology used in different contexts:

- **"Row"** - When working with spreadsheets/CSV files
- **"Sentence"** - When doing machine learning or NLP
- **"Record"** - When working with databases
- **"Observation"** - When doing statistical analysis

---

## Data Processing Context

**In Machine Learning:**
- **Tokens** → Input to tokenizers, text processing
- **Sentences/Rows** → Complete training examples
- **Features** → Input variables (X) and target variable (y)

**In Natural Language Processing:**
- **Tokens** → Words, punctuation, numbers
- **Sentences** → Complete text sequences
- **Features** → Word embeddings, TF-IDF scores, etc.
