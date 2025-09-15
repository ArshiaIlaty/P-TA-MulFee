import torch
import pandas as pd
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from utils import format_row  # assumes you have this in your utils.py

def load_prompts(csv_path, num_samples=None):
    df = pd.read_csv(csv_path)
    prompts = df.apply(format_row, axis=1).tolist()
    if num_samples:
        prompts = prompts[:num_samples]
    return prompts

def generate_responses(model, tokenizer, prompts, device="cuda", max_new_tokens=25, temperature=0.7):
    model.eval()
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(generated_text)
    return outputs

def main():
    csv_path = "heloc.csv"
    model_path = "./gpt2_ppo_hierarchical_heloc"
    output_csv = "ppo_generated_heloc.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer and PPO model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)

    print("Preparing prompts...")
    prompts = load_prompts(csv_path, num_samples=1000)  # change num_samples if desired

    print(f"Generating responses for {len(prompts)} prompts...")
    responses = generate_responses(model, tokenizer, prompts, device=device)

    print(f"Saving to {output_csv}...")
    df_out = pd.DataFrame({"prompt": prompts, "generated": responses})
    df_out.to_csv(output_csv, index=False)

    print("✅ Done.")

if __name__ == "__main__":
    main()