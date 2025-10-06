import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
from utils import format_row
import numpy as np
import logging
import argparse
import json
import os
from datetime import datetime
import sys
import gc
from transformers.utils import logging as hf_logging
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
hf_logging.set_verbosity_error()
import torch
print("Using device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
# Create logs directory
os.makedirs("dpo_training_logs", exist_ok=True)

class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs"""
    def __init__(self, prompts, chosen_responses, rejected_responses, tokenizer, max_length=512):
        self.prompts = prompts
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return min(len(self.prompts), len(self.chosen_responses), len(self.rejected_responses))
    
    def __getitem__(self, idx):
        # Safety check to prevent index out of range
        if idx >= len(self.prompts) or idx >= len(self.chosen_responses) or idx >= len(self.rejected_responses):
            raise IndexError(f"Index {idx} out of range. Prompts: {len(self.prompts)}, Chosen: {len(self.chosen_responses)}, Rejected: {len(self.rejected_responses)}")
        
        prompt = self.prompts[idx]
        chosen = self.chosen_responses[idx]
        rejected = self.rejected_responses[idx]
        
        # Tokenize prompt + chosen response
        chosen_text = prompt + " " + chosen
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize prompt + rejected response
        rejected_text = prompt + " " + rejected
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
        }

def robust_save(model, tokenizer, save_path, logger, label="final"):
    try:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer saved to {save_path} [{label}]")
        print(f"Model and tokenizer saved to {save_path} [{label}]")
    except Exception as e:
        logger.error(f"Failed to save model [{label}]: {e}")
        print(f"Failed to save model [{label}]: {e}")

def hierarchical_reward_fn(hierarchical_discriminators, generated_text):
    """Calculate hierarchical reward for generated text"""
    feedback = hierarchical_discriminators.get_multi_level_feedback(generated_text)
    reward = (
        feedback['token'] * 0.2 +
        feedback['sentence'] * 0.3 +
        feedback['row'] * 0.3 +
        np.mean(list(feedback['features'].values())) * 0.2
    )
    return reward, feedback

def generate_preference_pairs(model, tokenizer, prompts, hierarchical_discriminators, device, num_pairs=4):
    """Generate preference pairs (chosen vs rejected) for DPO training"""
    chosen_responses = []
    rejected_responses = []
    chosen_rewards = []
    rejected_rewards = []
    reward_breakdowns = []
    
    # Set model to eval mode for generation
    model.eval()
    
    for prompt in prompts:
        # Generate multiple responses for the same prompt
        responses = []
        rewards = []
        
        # Generate 4 different responses with different temperatures for diversity
        temperatures = [0.6, 0.8, 1.0, 1.2]
        for temp in temperatures:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                response_ids = model.generate(
                    input_ids,
                    max_new_tokens=25,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                    return_dict_in_generate=False,
                    repetition_penalty=1.1
                )
            
            # Decode the full response
            if response_ids.dim() > 1:
                response_ids = response_ids.squeeze(0)
            full_response = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
            responses.append(full_response)
            
            # Calculate reward
            reward, feedback = hierarchical_reward_fn(hierarchical_discriminators, full_response)
            rewards.append(reward)
        
        # Select the best and worst responses based on rewards
        best_idx = np.argmax(rewards)
        worst_idx = np.argmin(rewards)
        
        # Only create preference pair if there's a meaningful difference
        if rewards[best_idx] > rewards[worst_idx] + 0.01:  # Minimum difference threshold
            chosen_responses.append(responses[best_idx])
            rejected_responses.append(responses[worst_idx])
            chosen_rewards.append(rewards[best_idx])
            rejected_rewards.append(rewards[worst_idx])
            
            # Store reward breakdown for the chosen response
            _, feedback = hierarchical_reward_fn(hierarchical_discriminators, responses[best_idx])
            reward_breakdowns.append({
                "token": float(feedback['token']),
                "sentence": float(feedback['sentence']),
                "row": float(feedback['row']),
                "features_mean": float(np.mean(list(feedback['features'].values())))
            })
    
    # Set model back to training mode
    model.train()
    
    return chosen_responses, rejected_responses, chosen_rewards, rejected_rewards, reward_breakdowns

def compute_dpo_loss(model, chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask, beta=0.1):
    """Compute DPO loss for single sample"""
    # Get logits for chosen response
    chosen_outputs = model(
        input_ids=chosen_input_ids,
        attention_mask=chosen_attention_mask
    )
    chosen_logits = chosen_outputs.logits
    
    # Get logits for rejected response
    rejected_outputs = model(
        input_ids=rejected_input_ids,
        attention_mask=rejected_attention_mask
    )
    rejected_logits = rejected_outputs.logits
    
    # Compute log probabilities
    chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
    
    # Compute mean log probability across the sequence
    chosen_log_probs_mean = chosen_log_probs.mean(dim=-1)
    rejected_log_probs_mean = rejected_log_probs.mean(dim=-1)
    
    # DPO loss
    loss = -torch.nn.functional.logsigmoid(
        beta * (chosen_log_probs_mean - rejected_log_probs_mean)
    )
    
    return loss.mean()

def save_training_metrics(step, chosen_rewards, rejected_rewards, reward_breakdowns, loss, gpu_memory_allocated, gpu_memory_reserved, save_dir="dpo_training_logs"):
    """Save training metrics to JSON files"""
    metrics = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "loss": float(loss),
        "mean_chosen_reward": float(np.mean(chosen_rewards)),
        "mean_rejected_reward": float(np.mean(rejected_rewards)),
        "reward_gap": float(np.mean(chosen_rewards) - np.mean(rejected_rewards)),
        "std_chosen_reward": float(np.std(chosen_rewards)),
        "std_rejected_reward": float(np.std(rejected_rewards)),
        "min_chosen_reward": float(np.min(chosen_rewards)),
        "max_chosen_reward": float(np.max(chosen_rewards)),
        "min_rejected_reward": float(np.min(rejected_rewards)),
        "max_rejected_reward": float(np.max(rejected_rewards)),
        "gpu_memory_allocated_gb": float(gpu_memory_allocated),
        "gpu_memory_reserved_gb": float(gpu_memory_reserved),
        "reward_breakdowns": reward_breakdowns,
        "avg_reward_components": {
            "token": float(np.mean([rb['token'] for rb in reward_breakdowns])),
            "sentence": float(np.mean([rb['sentence'] for rb in reward_breakdowns])),
            "row": float(np.mean([rb['row'] for rb in reward_breakdowns])),
            "features_mean": float(np.mean([rb['features_mean'] for rb in reward_breakdowns]))
        }
    }
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/step_{step:04d}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(f"{save_dir}/training_summary.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception as e:
        print(f"Failed to save training metrics at step {step}: {e}")
    return metrics

def save_checkpoint(model, tokenizer, save_path, step, logger):
    """Save model checkpoint"""
    checkpoint_path = f"{save_path}_step{step}"
    print(f"Saving checkpoint at step {step} to {checkpoint_path}")
    robust_save(model, tokenizer, checkpoint_path, logger, label=f"checkpoint step {step}")

def main(
    csv_path="heloc.csv",
    model_name="gpt2",
    save_path="./gpt2_dpo_hierarchical_heloc",
    total_steps=1000,
    batch_size=1,  # Reduced batch size for memory
    learning_rate=1e-5,
    device="cuda",
    checkpoint_interval=100,
    beta=0.1
):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("dpo_training_logs/dpo_training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Test save at the start
    try:
        os.makedirs("dpo_training_logs/test_save", exist_ok=True)
        with open("dpo_training_logs/test_save/test.txt", "w") as f:
            f.write("test")
        os.remove("dpo_training_logs/test_save/test.txt")
        os.rmdir("dpo_training_logs/test_save")
        print("Test save succeeded.")
    except Exception as e:
        print(f"Test save failed: {e}")
        sys.exit(1)

    # Log training configuration
    config = {
        "csv_path": csv_path,
        "model_name": model_name,
        "save_path": save_path,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "checkpoint_interval": checkpoint_interval,
        "beta": beta,
        "start_time": datetime.now().isoformat()
    }
    try:
        with open("dpo_training_logs/training_config.json", "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Failed to save training config: {e}")
    logger.info(f"Starting DPO training with config: {config}")

    # Load data and prompts
    df = pd.read_csv(csv_path)
    prompts = df.apply(format_row, axis=1).tolist()
    logger.info(f"Loaded {len(prompts)} prompts from {csv_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set device properly
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Memory optimization
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
    
    model.train()

    # Load hierarchical discriminator system
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(model_name=model_name, device=device)
    
    print(f"Using device: {device}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Gradient accumulation settings
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 steps
    optimizer.zero_grad()

    logger.info("Starting DPO training loop...")
    all_chosen_rewards = []
    all_rejected_rewards = []
    training_history = []
    
    try:
        for step in range(total_steps):
            print(f"Starting step {step}/{total_steps}")
            logger.info(f"Starting step {step}/{total_steps}")
            # Sample batch of prompts
            batch_prompts = np.random.choice(prompts, batch_size).tolist()
            
            # Generate preference pairs (reduced for memory efficiency)
            chosen_responses, rejected_responses, chosen_rewards, rejected_rewards, reward_breakdowns = generate_preference_pairs(
                model, tokenizer, batch_prompts, hierarchical_discriminators, device, num_pairs=2
            )
            
            if len(chosen_responses) == 0:
                logger.warning(f"Step {step}: No valid preference pairs generated, skipping...")
                continue
            
            # Create DPO dataset
            logger.info(f"Creating dataset with {len(batch_prompts)} prompts, {len(chosen_responses)} chosen, {len(rejected_responses)} rejected")
            dpo_dataset = DPODataset(
                batch_prompts, chosen_responses, rejected_responses, tokenizer
            )
            
            # Create dataloader
            dataloader = DataLoader(dpo_dataset, batch_size=len(dpo_dataset), shuffle=False)
            
            # Training step - process one sample at a time to save memory
            total_loss = 0.0
            num_samples = 0
            accumulated_loss = 0.0
            
            for batch in dataloader:
                # Move batch to device
                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                
                # Process each sample individually to save memory
                actual_batch_size = chosen_input_ids.size(0)
                for i in range(actual_batch_size):
                    # Extract single sample
                    chosen_ids = chosen_input_ids[i:i+1]
                    chosen_mask = chosen_attention_mask[i:i+1]
                    rejected_ids = rejected_input_ids[i:i+1]
                    rejected_mask = rejected_attention_mask[i:i+1]
                    
                    # Compute DPO loss for single sample
                    loss = compute_dpo_loss(
                        model, chosen_ids, chosen_mask, 
                        rejected_ids, rejected_mask, beta
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                    num_samples += 1
                    
                    # Update weights every gradient_accumulation_steps
                    if num_samples % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += accumulated_loss
                        accumulated_loss = 0.0
                    
                    # Clear cache after each sample
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Force memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            # Handle remaining gradients
            if accumulated_loss > 0:
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accumulated_loss
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Log metrics
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            
            metrics = save_training_metrics(
                step, chosen_rewards, rejected_rewards, reward_breakdowns, 
                total_loss, gpu_memory_allocated, gpu_memory_reserved
            )
            
            all_chosen_rewards.extend(chosen_rewards)
            all_rejected_rewards.extend(rejected_rewards)
            training_history.append(metrics)
            
            if step % checkpoint_interval == 0 and step > 0:
                save_checkpoint(model, tokenizer, save_path, step, logger)
                
            # Log every step for better monitoring
            logger.info(f"Step {step}: loss = {total_loss:.4f}, "
                       f"mean chosen reward = {np.mean(chosen_rewards):.4f}, "
                       f"mean rejected reward = {np.mean(rejected_rewards):.4f}, "
                       f"reward gap = {np.mean(chosen_rewards) - np.mean(rejected_rewards):.4f}, "
                       f"overall mean chosen = {np.mean(all_chosen_rewards):.4f}, "
                       f"memory = {gpu_memory_allocated:.2f}GB")
        
        # Final summary
        final_summary = {
            "total_steps": total_steps,
            "final_mean_chosen_reward": float(np.mean(all_chosen_rewards)),
            "final_mean_rejected_reward": float(np.mean(all_rejected_rewards)),
            "final_reward_gap": float(np.mean(all_chosen_rewards) - np.mean(all_rejected_rewards)),
            "chosen_reward_trend": all_chosen_rewards,
            "rejected_reward_trend": all_rejected_rewards,
            "end_time": datetime.now().isoformat()
        }
        try:
            with open("dpo_training_logs/final_summary.json", "w") as f:
                json.dump(final_summary, f, indent=2)
        except Exception as e:
            print(f"Failed to save final summary: {e}")
            
        print("Training complete. Attempting to save final model...")
        robust_save(model, tokenizer, save_path, logger, label="final")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Final save attempt in finally block...")
        robust_save(model, tokenizer, save_path, logger, label="final-finally")
        print("If you see this message, the save attempt was made in the finally block.")
        logger.info("If you see this message, the save attempt was made in the finally block.")
        print(f"Check {save_path} and dpo_training_logs/ for outputs.")
        logger.info(f"Check {save_path} and dpo_training_logs/ for outputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO training for HELOC with hierarchical discriminator preference generation.")
    parser.add_argument('--csv_path', type=str, default='heloc.csv', help='Path to HELOC CSV file')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model name or path')
    parser.add_argument('--save_path', type=str, default='./gpt2_dpo_hierarchical_heloc', help='Where to save DPO-finetuned model')
    parser.add_argument('--total_steps', type=int, default=1000, help='Number of DPO steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DPO')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for DPO')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_interval', type=int, default=25, help='Save checkpoint every N steps')
    parser.add_argument('--beta', type=float, default=0.1, help='DPO beta parameter (controls preference strength)')
    
    args = parser.parse_args()
    main(
        csv_path=args.csv_path,
        model_name=args.model_name,
        save_path=args.save_path,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        beta=args.beta
    ) 