# P-TA: Privacy-Preserving Tabular Data Augmentation with LLMs

P-TA leverages large language models (LLMs) and hierarchical discriminators to generate high-quality, privacy-preserving synthetic tabular data. The project supports advanced pipelines for datasets like HELOC, with multi-level feedback, PPO-based reinforcement learning, and adversarial training.

---

## 🚀 Project Workflow Overview

```mermaid
graph TD
    A[Start: Prepare Real Data] --> B[Format Data as Prompts]
    B --> C[Train Model (GREAT or PPO)]
    C --> D[Generate Synthetic Data]
    D --> E[Parse & Clean Synthetic Data]
    E --> F[Evaluate Synthetic Data]
    F --> G[Compare with Baseline]
    G --> H[Iterate: Tune, Retrain, Regenerate]
```

---

## 🛠️ Step-by-Step Process

### 1. **Prepare Environment & Data**
- Clone the repo and install dependencies:
  ```bash
  git clone <repo-url>
  cd P-TA
  pip install torch==2.3.0 tokenizers==0.19.1 transformers==4.40.1 pandas==2.2.2 scikit-learn numpy trl
  ```
- Place your real dataset (e.g., `heloc.csv`) in the project root.

### 2. **Format Data as Prompts**
- Use the provided `format_row` utility to convert each row to a text prompt.

### 3. **Model Training**
- **Supervised (GREAT):**
  - Use `great_heloc.py` or `great.py` to fine-tune GPT-2 on your tabular data.
- **Reinforcement Learning (PPO):**
  - Use `ppo_hierarchical_heloc.py` to train with PPO and hierarchical discriminator feedback.
  - Example (on GPU 3):
    ```bash
    CUDA_VISIBLE_DEVICES=3 python ppo_hierarchical_heloc.py --total_steps 1000 --batch_size 2 --learning_rate 1e-5
    ```
  - The script will save the model to `gpt2_ppo_hierarchical_heloc/` and logs to `ppo_training_logs/`.

### 4. **Generate Synthetic Data**
- Use `generate_ppo_heloc.py` to generate synthetic data from your PPO-finetuned model:
  ```bash
  CUDA_VISIBLE_DEVICES=3 python generate_ppo_heloc.py
  ```
- The script uses only the first 3 features as prompt for diversity and saves results to `output_ppo_heloc.csv`.

### 5. **Parse & Clean Synthetic Data**
- Parse each generated row into structured columns (see parsing script suggestion below).
- Remove incomplete or malformed rows.

### 6. **Evaluate Synthetic Data**
- Use your evaluation script (e.g., `evaluate_heloc.py`) to assess utility, privacy, and diversity:
  ```bash
  python evaluate_heloc.py --synthetic_csv output_ppo_heloc_parsed.csv
  ```
- Compare results to baseline and previous models.

### 7. **Iterate & Tune**
- Adjust PPO training steps, batch size, learning rate, or prompt length for better results.
- Retrain and regenerate as needed.

---

## 📂 Major Scripts & Their Purpose

| Script                        | Purpose                                                      |
|-------------------------------|--------------------------------------------------------------|
| `great_heloc.py` / `great.py` | Supervised fine-tuning (GREAT) of GPT-2 on tabular data      |
| `ppo_hierarchical_heloc.py`   | PPO-based RL training with hierarchical discriminator reward  |
| `generate_ppo_heloc.py`       | Generate synthetic data from PPO-finetuned model             |
| `evaluate_heloc.py`           | Evaluate synthetic data utility, privacy, and diversity      |
| `utils.py`                    | Data formatting and helper functions                         |

---

## ⚡ Best Practices
- **GPU Selection:** Use `CUDA_VISIBLE_DEVICES=<gpu_id>` to run on a specific GPU.
- **Batch Size:** Start with 1–2 for PPO; increase if memory allows.
- **Prompt Length:** Use partial prompts (first 3–5 features) for more diverse generation.
- **Training Steps:** More steps = better generation. Start with 1000+ for PPO.
- **Monitoring:** Use `nvidia-smi` to monitor GPU usage. Check logs for OOM or instability.
- **Parsing:** Always parse and clean generated data before evaluation.
- **Evaluation:** Use the same metrics and scripts for all models for fair comparison.

---

## 🧩 Example: Parsing Synthetic Data
```python
import pandas as pd

def parse_generated_text_to_dict(text):
    fields = text.split(", ")
    data_dict = {}
    for field in fields:
        if " is " in field:
            key, value = field.split(" is ", 1)
            data_dict[key.strip()] = value.strip()
    return data_dict

df = pd.read_csv("output_ppo_heloc.csv")
parsed = [parse_generated_text_to_dict(row['synthetic_text']) for _, row in df.iterrows()]
df_parsed = pd.DataFrame(parsed)
df_parsed.to_csv("output_ppo_heloc_parsed.csv", index=False)
print("Saved parsed synthetic data to output_ppo_heloc_parsed.csv")
```

---

## 🌟 Project Highlights
- **Supports both supervised (GREAT) and RL (PPO) synthetic data generation**
- **Hierarchical discriminators for quality feedback**
- **Robust logging, checkpointing, and GPU support**
- **Flexible prompt and generation strategies**
- **Comprehensive evaluation pipeline**

---

## 📈 Iterative Improvement
- Train longer, use more diverse prompts, and tune hyperparameters for best results.
- Always parse, clean, and evaluate synthetic data before use.
- Compare against real data and previous models for continuous improvement.

---

For any questions or to contribute, please open an issue or pull request!
