import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from utils import format_row
import gc

# Force CUDA to only see GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Print available GPUs
print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set environment variables for better memory management
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:512"  # Limit memory allocation
)
torch.cuda.empty_cache()  # Clear GPU cache
gc.collect()  # Clear Python garbage collector

# Enable gradient checkpointing globally
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class GreatTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def train_great(
    csv_path="diabetes.csv", model_name="gpt2", save_pth="./gpt2_finetuned_diabetes"
):
    device = torch.device("cuda:0")  # Will be GPU 1 due to CUDA_VISIBLE_DEVICES
    print(f"Using device: {device}")

    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_texts = train_df.apply(format_row, axis=1).tolist()
    val_texts = val_df.apply(format_row, axis=1).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=64
        )  # Reduced sequence length

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    tokenized_datasets = DatasetDict(
        {
            "train": train_dataset.map(tokenize_function, batched=True, batch_size=500),
            "validation": val_dataset.map(
                tokenize_function, batched=True, batch_size=500
            ),
        }
    )

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned_diabetes",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        per_device_train_batch_size=2,  # Reduced batch size
        per_device_eval_batch_size=2,  # Reduced batch size
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs_diabetes",
        logging_steps=10,
        gradient_accumulation_steps=16,  # Added gradient accumulation
        fp16=True,  # Enable mixed precision
        gradient_checkpointing=True,  # Enable gradient checkpointing
        dataloader_num_workers=0,  # Disable multiprocessing
        no_cuda=False,
        local_rank=-1,
    )

    trainer = GreatTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    trainer.can_return_loss = True
    trainer.train()

    model.save_pretrained(save_pth)
    tokenizer.save_pretrained(save_pth)
