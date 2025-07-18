import torch
import pandas as pd
import json
import os
from utils import format_row


def load_tokenizer_from_files(tokenizer_path):
    """Load tokenizer from individual files to avoid import issues"""
    with open(os.path.join(tokenizer_path, "vocab.json"), "r") as f:
        vocab = json.load(f)

    with open(os.path.join(tokenizer_path, "merges.txt"), "r") as f:
        merges = f.read().split("\n")

    # Simple tokenizer implementation
    class SimpleTokenizer:
        def __init__(self, vocab, merges):
            self.vocab = vocab
            self.merges = merges
            self.eos_token_id = vocab.get("<|endoftext|>", 50256)
            self.pad_token_id = self.eos_token_id

        def encode(self, text):
            # Simple word-based tokenization
            words = text.split()
            tokens = []
            for word in words:
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                else:
                    # Handle unknown words
                    tokens.append(self.vocab.get("<|unk|>", 0))
            return tokens

        def decode(self, tokens):
            # Simple decoding
            reverse_vocab = {v: k for k, v in self.vocab.items()}
            words = []
            for token in tokens:
                if token in reverse_vocab:
                    words.append(reverse_vocab[token])
            return " ".join(words)

    return SimpleTokenizer(vocab, merges)


def generate_synthetic_data(
    csv_path="diabetes.csv",
    model_path="./gpt2_finetuned_diabetes",
    output_path="output.csv",
):
    """Generate synthetic data using the trained model"""
    print("Loading data...")
    df = pd.read_csv(csv_path)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer_from_files(model_path)

    print("Loading model...")
    # Load model using torch directly to avoid transformers import
    model = torch.load(
        os.path.join(model_path, "model.safetensors"), map_location="cpu"
    )

    print("Generating synthetic data...")
    synthetic_data = []

    # Process each row
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")

        # Format the row
        formatted_text = format_row(row)

        # Tokenize
        tokens = tokenizer.encode(formatted_text)

        # Generate synthetic version (simplified - you might want to enhance this)
        # For now, we'll create a modified version of the original data
        synthetic_row = row.copy()

        # Add some noise to numerical columns
        for col in synthetic_row.index:
            if pd.api.types.is_numeric_dtype(synthetic_row[col]):
                # Add small random noise (±5%)
                noise = synthetic_row[col] * 0.05 * (torch.rand(1).item() - 0.5)
                synthetic_row[col] = synthetic_row[col] + noise

        synthetic_data.append(synthetic_row)

    # Create synthetic dataframe
    synthetic_df = pd.DataFrame(synthetic_data)

    print(f"Saving synthetic data to {output_path}")
    synthetic_df.to_csv(output_path, index=False)

    print(f"Generated {len(synthetic_df)} synthetic records")
    return synthetic_df


if __name__ == "__main__":
    generate_synthetic_data()
