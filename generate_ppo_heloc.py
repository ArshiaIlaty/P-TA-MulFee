import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import format_row

MODEL_PATH = "./gpt2_ppo_hierarchical_heloc"
DATA_PATH = "heloc.csv"
OUTPUT_PATH = "output_ppo_heloc.csv"

# Load PPO model and tokenizer
print(f"Loading PPO model from {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load real data and format as prompts
print(f"Loading real data from {DATA_PATH}...")
df_real = pd.read_csv(DATA_PATH)
real_texts = df_real.apply(format_row, axis=1).tolist()

synthetic_texts = []
print(f"Generating synthetic data for {len(real_texts)} prompts (using only first 3 features as prompt)...")
for i, full_prompt in enumerate(real_texts):
    # Use only the first 3 features as prompt
    partial_prompt = ", ".join(full_prompt.split(", ")[:3])
    input_ids = tokenizer.encode(partial_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    synthetic_texts.append(generated_text)
    if i < 5:
        print(f"Prompt: {partial_prompt}")
        print(f"Generated: {generated_text}\n")
    if (i + 1) % 100 == 0 or (i + 1) == len(real_texts):
        print(f"Generated {i + 1}/{len(real_texts)} samples...")

# Save to CSV
pd.DataFrame({"synthetic_text": synthetic_texts}).to_csv(OUTPUT_PATH, index=False)
print(f"Saved synthetic data to {OUTPUT_PATH}") 