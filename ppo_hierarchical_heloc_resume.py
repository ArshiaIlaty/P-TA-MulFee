import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
import argparse
from datetime import datetime

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_save(model, tokenizer, save_path, logger, label="final"):
    """Robust model saving with error handling"""
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"✓ {label} model saved to {save_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save {label} model: {e}")
        return False

def hierarchical_reward_fn(hierarchical_discriminators, generated_text):
    """Calculate hierarchical reward for generated text"""
    try:
        # Use the correct method name
        feedback = hierarchical_discriminators.get_multi_level_feedback(generated_text)
        # Calculate overall reward as weighted average
        reward = (
            feedback['token'] * 0.25 + 
            feedback['sentence'] * 0.25 + 
            feedback['row'] * 0.25 + 
            np.mean(list(feedback['features'].values())) * 0.25
        )
        return reward, feedback
    except Exception as e:
        print(f"Error in reward calculation: {e}")
        return 0.0, {"token": 0.0, "sentence": 0.0, "row": 0.0, "features": {}}

def save_training_metrics(step, rewards, reward_breakdowns, gpu_memory_allocated, gpu_memory_reserved, save_dir="ppo_training_logs"):
    """Save training metrics to JSONL file"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate statistics
    rewards_array = [r.item() if torch.is_tensor(r) else r for r in rewards]
    mean_reward = np.mean(rewards_array)
    std_reward = np.std(rewards_array)
    min_reward = np.min(rewards_array)
    max_reward = np.max(rewards_array)
    
    # Calculate average reward components
    avg_components = {}
    if reward_breakdowns:
        component_keys = ['token', 'sentence', 'row', 'features_mean']
        for key in component_keys:
            values = [bd[key] for bd in reward_breakdowns if key in bd]
            avg_components[key] = np.mean(values) if values else 0.0
    
    # Create metrics entry
    metrics = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "min_reward": float(min_reward),
        "max_reward": float(max_reward),
        "gpu_memory_allocated_gb": float(gpu_memory_allocated / 1024**3),
        "gpu_memory_reserved_gb": float(gpu_memory_reserved / 1024**3),
        "reward_breakdowns": reward_breakdowns,
        "avg_reward_components": avg_components
    }
    
    # Save to JSONL file
    with open(os.path.join(save_dir, "training_summary.jsonl"), "a") as f:
        f.write(json.dumps(metrics) + "\n")

def save_checkpoint(model, tokenizer, save_path, step, logger):
    """Save model checkpoint"""
    checkpoint_path = f"{save_path}_step{step}"
    print(f"Saving checkpoint at step {step} to {checkpoint_path}")
    robust_save(model, tokenizer, checkpoint_path, logger, label=f"checkpoint step {step}")

def load_checkpoint(checkpoint_path, logger, device="cuda"):
    """Load model from checkpoint"""
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # Load the model directly from the checkpoint path
        model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logger.info(f"✓ Checkpoint loaded successfully from {checkpoint_path}")
        return model, tokenizer, True
    except Exception as e:
        logger.error(f"✗ Failed to load checkpoint: {e}")
        return None, None, False

def format_row(row):
    """Format a row as a prompt"""
    features = []
    for col in row.index:
        if col != 'RiskPerformance':
            features.append(f"{col} is {row[col]},")
    return " ".join(features)

def main(
    csv_path="heloc.csv",
    model_name="gpt2",
    save_path="./gpt2_ppo_hierarchical_heloc",
    total_steps=1000,
    batch_size=8,  # Reduced from 16 to prevent CUDA OOM
    learning_rate=1e-5,  # Reduced from 3e-5
    device="cuda",
    checkpoint_interval=25,
    kl_penalty=0.05,
    resume_from_step=None,  # New parameter for resuming
    resume_checkpoint_path=None  # New parameter for checkpoint path
):
    # Save training configuration
    config = {
        "csv_path": csv_path,
        "model_name": model_name,
        "save_path": save_path,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "checkpoint_interval": checkpoint_interval,
        "kl_penalty": kl_penalty,
        "resume_from_step": resume_from_step,
        "resume_checkpoint_path": resume_checkpoint_path,
        "start_time": datetime.now().isoformat()
    }
    
    os.makedirs("ppo_training_logs", exist_ok=True)
    with open("ppo_training_logs/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration: {config}")
    
    # Load data and prompts
    df = pd.read_csv(csv_path)
    prompts = df.apply(format_row, axis=1).tolist()
    logger.info(f"Loaded {len(prompts)} prompts from {csv_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from_step is not None and resume_checkpoint_path is not None:
        model, tokenizer, success = load_checkpoint(resume_checkpoint_path, logger, device)
        if success:
            start_step = resume_from_step
            logger.info(f"Resuming training from step {start_step}")
        else:
            logger.warning("Failed to load checkpoint, starting from scratch")
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
    else:
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
        for step in range(start_step, total_steps):
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
            if step == start_step:
                logger.info(f"Query tensor shapes: {[q.shape for q in query_tensors]}")
                logger.info(f"Response tensor shapes: {[r.shape for r in response_tensors]}")
                logger.info(f"Reward types: {[type(r) for r in rewards]}")
            
            # PPO training step
            try:
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                logger.info(f"Step {step}: PPO stats - {stats}")
            except Exception as e:
                logger.error(f"Error in PPO step {step}: {e}")
                continue
            
            # Calculate and log metrics
            mean_reward = torch.stack(rewards).mean().item()
            all_rewards.append(mean_reward)
            
            # GPU memory monitoring
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_reserved = torch.cuda.memory_reserved()
            else:
                gpu_memory_allocated = 0
                gpu_memory_reserved = 0
            
            # Save training metrics
            save_training_metrics(step, rewards, reward_breakdowns, gpu_memory_allocated, gpu_memory_reserved)
            
            # Log progress
            if step % 10 == 0:
                logger.info(f"Step {step}/{total_steps}: Mean reward = {mean_reward:.4f}")
            
            # Save checkpoint
            if step % checkpoint_interval == 0 and step > 0:
                save_checkpoint(model, tokenizer, save_path, step, logger)
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    # Save final model
    logger.info("Training completed. Saving final model...")
    robust_save(model, tokenizer, save_path, logger, label="final")
    
    # Save training history
    training_history = {
        "all_rewards": all_rewards,
        "final_mean_reward": np.mean(all_rewards[-100:]) if all_rewards else 0.0,
        "best_reward": max(all_rewards) if all_rewards else 0.0
    }
    
    with open("ppo_training_logs/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training completed. Final mean reward: {training_history['final_mean_reward']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training with Hierarchical Discriminators")
    parser.add_argument('--csv_path', type=str, default="heloc.csv", help='Path to CSV file')
    parser.add_argument('--model_name', type=str, default="gpt2", help='Model name')
    parser.add_argument('--save_path', type=str, default="./gpt2_ppo_hierarchical_heloc", help='Save path')
    parser.add_argument('--total_steps', type=int, default=5000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda", help='Device')
    parser.add_argument('--checkpoint_interval', type=int, default=25, help='Save checkpoint every N steps')
    parser.add_argument('--kl_penalty', type=float, default=0.05, help='KL penalty')
    parser.add_argument('--resume_from_step', type=int, default=None, help='Resume from step number')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    logger.info(f"Using device: {device}")

    main(
        csv_path=args.csv_path,
        model_name=args.model_name,
        save_path=args.save_path,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_interval=args.checkpoint_interval,
        kl_penalty=args.kl_penalty,
        resume_from_step=args.resume_from_step,
        resume_checkpoint_path=args.resume_checkpoint_path
    ) 