import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
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
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
import torch
print("Using device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
# Create logs directory
os.makedirs("dpo_training_logs", exist_ok=True)

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
    
    for prompt in prompts:
        # Generate multiple responses for the same prompt
        responses = []
        rewards = []
        
        # Generate 4 different responses with different temperatures for diversity
        temperatures = [0.6, 0.8, 1.0, 1.2]
        for temp in temperatures:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            input_ids = inputs["input_ids"][0]
            
            with torch.no_grad():
                response_ids = model.generate(
                    input_ids,
                    max_new_tokens=25,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode the full response
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
    
    return chosen_responses, rejected_responses, chosen_rewards, rejected_rewards, reward_breakdowns

def save_training_metrics(step, chosen_rewards, rejected_rewards, reward_breakdowns, gpu_memory_allocated, gpu_memory_reserved, save_dir="dpo_training_logs"):
    """Save training metrics to JSON files"""
    metrics = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
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
    batch_size=4,
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
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.gradient_checkpointing_enable()
    model.eval()

    # Load hierarchical discriminator system
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(model_name=model_name, device=device)

    # DPO config
    dpo_config = DPOConfig(
        beta=beta,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        target_kl=0.1,
        max_prompt_length=512,
        max_length=512,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=checkpoint_interval,
        eval_steps=checkpoint_interval,
        output_dir=save_path,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Initialize DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=dpo_config,
        beta=beta,
        train_dataset=None,  # We'll provide data dynamically
        eval_dataset=None,
        tokenizer_name_or_path=model_name,
        max_length=512,
        max_prompt_length=512,
        max_target_length=25,
        padding=True,
        truncation=True,
    )

    logger.info("Starting DPO training loop...")
    all_chosen_rewards = []
    all_rejected_rewards = []
    training_history = []
    
    try:
        for step in range(total_steps):
            # Sample batch of prompts
            batch_prompts = np.random.choice(prompts, batch_size).tolist()
            
            # Generate preference pairs
            chosen_responses, rejected_responses, chosen_rewards, rejected_rewards, reward_breakdowns = generate_preference_pairs(
                model, tokenizer, batch_prompts, hierarchical_discriminators, device, num_pairs=4
            )
            
            if len(chosen_responses) == 0:
                logger.warning(f"Step {step}: No valid preference pairs generated, skipping...")
                continue
            
            # Create training data for DPO
            training_data = []
            for i in range(len(chosen_responses)):
                training_data.append({
                    "prompt": batch_prompts[i],
                    "chosen": chosen_responses[i],
                    "rejected": rejected_responses[i],
                    "chosen_reward": chosen_rewards[i],
                    "rejected_reward": rejected_rewards[i]
                })
            
            # Convert to DPO format
            dpo_batch = {
                "prompt": [item["prompt"] for item in training_data],
                "chosen": [item["chosen"] for item in training_data],
                "rejected": [item["rejected"] for item in training_data]
            }
            
            # Run DPO training step
            try:
                # Create a simple dataset for DPO training
                from datasets import Dataset
                
                # Prepare data in the format expected by DPOTrainer
                dpo_data = []
                for i in range(len(chosen_responses)):
                    dpo_data.append({
                        "prompt": batch_prompts[i],
                        "chosen": chosen_responses[i],
                        "rejected": rejected_responses[i]
                    })
                
                # Create dataset
                dataset = Dataset.from_list(dpo_data)
                
                # Set the dataset for this step
                dpo_trainer.train_dataset = dataset
                
                # Run DPO training step
                train_output = dpo_trainer.train()
                
                # Get loss from training output
                if hasattr(train_output, 'train_loss'):
                    loss = train_output.train_loss
                else:
                    loss = 0.0
                    
                logger.info(f"DPO step {step} loss: {loss}")
                
            except Exception as e:
                logger.error(f"DPO step failed: {e}")
                print(f"DPO step failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            # Log metrics
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            
            metrics = save_training_metrics(
                step, chosen_rewards, rejected_rewards, reward_breakdowns, 
                gpu_memory_allocated, gpu_memory_reserved
            )
            
            all_chosen_rewards.extend(chosen_rewards)
            all_rejected_rewards.extend(rejected_rewards)
            training_history.append(metrics)
            
            if step % checkpoint_interval == 0 and step > 0:
                save_checkpoint(model, tokenizer, save_path, step, logger)
                
            if step % 10 == 0:
                logger.info(f"Step {step}: mean chosen reward = {np.mean(chosen_rewards):.4f}, "
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
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save checkpoint every N steps')
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