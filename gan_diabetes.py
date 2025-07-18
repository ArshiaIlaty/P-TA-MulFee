import torch
import random
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AdamW,
)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from classifier_diabetes import remove_random_values


def gpt2_loss_function(input_text, missing_slots, classifier, tokenizer, model):
    tokens = input_text.split(", ")
    new_tokens = tokens[:]

    for idx, col_name in missing_slots.items():
        prompt = f"{col_name} is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 5,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_value = (
            generated_text.replace(f"{col_name} is", "").strip().split(",")[0]
        )
        new_tokens[idx] = f"{col_name} is {generated_value}"

    generated_text = ", ".join(new_tokens)

    encoding = tokenizer(
        generated_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

    with torch.no_grad():
        classifier_outputs = classifier(input_ids, attention_mask)
        logits = classifier_outputs
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    return pred


def train_gpt2_with_gan(df, model, tokenizer, classifier, num_epochs=3, N=2):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for _, row in df.iterrows():
            input_text = row["formatted_text"]

            corrupted_text, missing_slots = remove_random_values(
                input_text, num_remove=N
            )

            pred = gpt2_loss_function(
                corrupted_text, missing_slots, classifier, tokenizer, model
            )

            if pred == 0:
                optimizer = AdamW(model.parameters(), lr=5e-5)
                optimizer.zero_grad()

                # Compute GPT-2 loss
                input_ids = tokenizer(corrupted_text, return_tensors="pt").input_ids.to(
                    model.device
                )
                labels = input_ids.clone().to(model.device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(df)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")


def train_gpt2_with_hierarchical_gan(
    df, model, tokenizer, hierarchical_discriminators, device, num_epochs=1, N=2
):
    """
    Train GPT-2 generator with hierarchical discriminator feedback (adversarial GAN-style)
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        for idx, row in df.iterrows():
            input_text = (
                row["formatted_text"] if "formatted_text" in row else format_row(row)
            )
            # Corrupt input (optional, or just use as prompt)
            corrupted_text, missing_slots = remove_random_values(
                input_text, num_remove=N
            )
            prompt = corrupted_text
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Generate synthetic text
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 20,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                )
            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # Get hierarchical feedback
            feedback = hierarchical_discriminators.get_multi_level_feedback(
                generated_text
            )
            # Composite loss (the lower the feedback, the more "fake"; so use 1-feedback)
            composite_score = (
                0.2 * feedback["token"]
                + 0.3 * feedback["sentence"]
                + 0.3 * feedback["row"]
                + 0.2 * sum(feedback["features"].values()) / len(feedback["features"])
            )
            adv_loss = (
                1.0 - composite_score
            )  # Encourage generator to maximize discriminator output
            adv_loss = torch.tensor(adv_loss, requires_grad=True, device=device)

            # Standard language modeling loss (optional, can combine)
            lm_input_ids = tokenizer(generated_text, return_tensors="pt").input_ids.to(
                device
            )
            labels = lm_input_ids.clone()
            outputs = model(lm_input_ids, labels=labels)
            lm_loss = outputs.loss

            # Total loss: combine adversarial and LM loss
            total_gen_loss = lm_loss + adv_loss

            optimizer.zero_grad()
            total_gen_loss.backward()
            optimizer.step()

            total_loss += total_gen_loss.item()

            if idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}, Sample {idx}, Adv Loss: {adv_loss.item():.4f}, LM Loss: {lm_loss.item():.4f}, Total Loss: {total_gen_loss.item():.4f}"
                )

            # Optionally: update discriminators here (not implemented, as discriminators are trained separately)
            # hierarchical_discriminators.train_discriminators(...)
        avg_loss = total_loss / len(df)
        print(
            f"[Hierarchical GAN] Epoch {epoch+1}/{num_epochs}, Average Total Loss: {avg_loss:.4f}"
        )
