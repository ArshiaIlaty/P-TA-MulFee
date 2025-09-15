import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
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
os.makedirs("ppo_training_logs", exist_ok=True)

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
    feedback = hierarchical_discriminators.get_multi_level_feedback(generated_text)
    reward = (
        feedback['token'] * 0.2 +
        feedback['sentence'] * 0.3 +
        feedback['row'] * 0.3 +
        np.mean(list(feedback['features'].values())) * 0.2
    )
    return reward, feedback

def save_training_metrics(step, rewards, reward_breakdowns, gpu_memory_allocated, gpu_memory_reserved, save_dir="ppo_training_logs"):
    metrics = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
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
    checkpoint_path = f"{save_path}_step{step}"
    print(f"Saving checkpoint at step {step} to {checkpoint_path}")
    robust_save(model, tokenizer, checkpoint_path, logger, label=f"checkpoint step {step}")

def main(
    csv_path="heloc.csv",
    model_name="gpt2",
    save_path="./gpt2_ppo_hierarchical_heloc",
    total_steps=1000,
    batch_size=4,
    learning_rate=1e-5,
    device="cuda",
    checkpoint_interval=100,
    kl_penalty=0.1
):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("ppo_training_logs/ppo_training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Test save at the start
    try:
        os.makedirs("ppo_training_logs/test_save", exist_ok=True)
        with open("ppo_training_logs/test_save/test.txt", "w") as f:
            f.write("test")
        os.remove("ppo_training_logs/test_save/test.txt")
        os.rmdir("ppo_training_logs/test_save")
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
        "kl_penalty": kl_penalty,
        "start_time": datetime.now().isoformat()
    }
    try:
        with open("ppo_training_logs/training_config.json", "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Failed to save training config: {e}")
    logger.info(f"Starting PPO training with config: {config}")

    # Load data and prompts
    df = pd.read_csv(csv_path)
    prompts = df.apply(format_row, axis=1).tolist()
    logger.info(f"Loaded {len(prompts)} prompts from {csv_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
    model.gradient_checkpointing_enable()
    model.eval()

    # Load hierarchical discriminator system
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(model_name=model_name, device=device)

    # PPO config
    ppo_config = PPOConfig(
        batch_size=batch_size,
        mini_batch_size=batch_size,
        learning_rate=learning_rate,
    )
    try:
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer
        )
    except TypeError:
        ppo_trainer = PPOTrainer(model, tokenizer)
        ppo_trainer.batch_size = batch_size
        ppo_trainer.mini_batch_size = batch_size
        ppo_trainer.learning_rate = learning_rate
    
    if hasattr(ppo_trainer, 'kl_ctl'):
        ppo_trainer.kl_ctl.value = kl_penalty
        logger.info(f"Set KL penalty to {kl_penalty}")

    logger.info("Starting PPO training loop...")
    all_rewards = []
    training_history = []
    
    try:
        for step in range(total_steps):
            batch_prompts = np.random.choice(prompts, batch_size).tolist()
            responses = []
            rewards = []
            query_tensors = []
            response_tensors = []
            reward_breakdowns = []
            
            for prompt in batch_prompts:
                # Tokenize the prompt
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                input_ids = inputs["input_ids"][0]  # Get first (and only) sequence
                
                # Generate response using PPO trainer - don't pass attention_mask as separate parameter
                with torch.no_grad():
                    response_ids = ppo_trainer.generate(
                        query_tensor=input_ids,
                        max_new_tokens=25,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Convert response_ids to tensor if it's a list
                if isinstance(response_ids, list):
                    response_ids = torch.tensor(response_ids, dtype=torch.long)
                elif response_ids.dim() > 1:
                    response_ids = response_ids.squeeze(0)
                
                # Decode the full response
                full_response = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
                responses.append(full_response)
                
                # Calculate reward
                reward, feedback = hierarchical_reward_fn(hierarchical_discriminators, full_response)
                rewards.append(torch.tensor(reward, dtype=torch.float32))  # Convert to tensor
                
                reward_breakdowns.append({
                    "token": float(feedback['token']),
                    "sentence": float(feedback['sentence']),
                    "row": float(feedback['row']),
                    "features_mean": float(np.mean(list(feedback['features'].values())))
                })
                
                # Store tensors for PPO training
                # Make sure both query and response are 1D tensors
                query_tensors.append(input_ids.squeeze().detach().cpu())
                
                # Extract only the generated part (response after the prompt)
                prompt_length = len(input_ids)
                if len(response_ids) > prompt_length:
                    generated_part = response_ids[prompt_length:].detach().cpu()
                else:
                    # If no new tokens were generated, create a minimal response
                    generated_part = torch.tensor([tokenizer.pad_token_id], dtype=torch.long)
                response_tensors.append(generated_part)
                
            # Debug: print shapes and types
            print(f"Step {step} - Tensor shapes:")
            for i, (q, r) in enumerate(zip(query_tensors, response_tensors)):
                print(f"  Query {i}: {q.shape}, Response {i}: {r.shape}")
                print(f"  Query type: {type(q)}, Response type: {type(r)}")
            
            print(f"Rewards: {[r.item() if torch.is_tensor(r) else r for r in rewards]}")
            
            # Ensure all tensors are 1D and rewards are tensors
            query_tensors = [q.squeeze() if q.dim() > 1 else q for q in query_tensors]
            response_tensors = [r.squeeze() if r.dim() > 1 else r for r in response_tensors]
            
            # Convert rewards to tensors if they aren't already
            rewards = [torch.tensor(r, dtype=torch.float32) if not torch.is_tensor(r) else r for r in rewards]
            
            # Run PPO step
            try:
                ppo_trainer.step(query_tensors, response_tensors, rewards)
            except Exception as e:
                print(f"PPO step failed: {e}")
                print("Tensor info:")
                for i, (q, r, rew) in enumerate(zip(query_tensors, response_tensors, rewards)):
                    print(f"  Item {i}: query {q.shape}, response {r.shape}, reward {rew}")
                raise e
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            # Log metrics
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            
            # Convert tensor rewards back to float for logging
            reward_values = [r.item() if torch.is_tensor(r) else float(r) for r in rewards]
            metrics = save_training_metrics(step, reward_values, reward_breakdowns, gpu_memory_allocated, gpu_memory_reserved)
            all_rewards.extend(reward_values)
            training_history.append(metrics)
            
            if step % checkpoint_interval == 0 and step > 0:
                save_checkpoint(model, tokenizer, save_path, step, logger)
                
            if step % 10 == 0:
                logger.info(f"Step {step}: mean reward = {np.mean(reward_values):.4f}, "
                           f"overall mean = {np.mean(all_rewards):.4f}, "
                           f"memory = {gpu_memory_allocated:.2f}GB")
        
        # Final summary
        final_summary = {
            "total_steps": total_steps,
            "final_mean_reward": float(np.mean(all_rewards)),
            "final_std_reward": float(np.std(all_rewards)),
            "reward_trend": all_rewards,
            "end_time": datetime.now().isoformat()
        }
        try:
            with open("ppo_training_logs/final_summary.json", "w") as f:
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
        print(f"Check {save_path} and ppo_training_logs/ for outputs.")
        logger.info(f"Check {save_path} and ppo_training_logs/ for outputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO RL training for HELOC with hierarchical discriminator reward.")
    parser.add_argument('--csv_path', type=str, default='heloc.csv', help='Path to HELOC CSV file')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model name or path')
    parser.add_argument('--save_path', type=str, default='./gpt2_ppo_hierarchical_heloc', help='Where to save PPO-finetuned model')
    parser.add_argument('--total_steps', type=int, default=1000, help='Number of PPO steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for PPO')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for PPO')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save checkpoint every N steps')
    parser.add_argument('--kl_penalty', type=float, default=0.1, help='KL divergence penalty (lower = less constraint)')
    
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
        kl_penalty=args.kl_penalty
    )