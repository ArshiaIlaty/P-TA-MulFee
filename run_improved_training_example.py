import torch
import pandas as pd
import numpy as np
import logging
from improved_discriminator_training import ImprovedDiscriminatorTraining
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import format_row
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data_simple(model, tokenizer, real_texts, num_samples=100, device="cuda"):
    """Simple synthetic data generation for demonstration"""
    synthetic_texts = []
    
    for i in range(min(num_samples, len(real_texts))):
        text = real_texts[i]
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        synthetic_texts.append(generated_text)
    
    return synthetic_texts

def main():
    """Working example of improved discriminator training"""
    logger.info("Starting improved discriminator training example...")
    
    # Initialize wandb
    try:
        wandb.init(project="improved-discriminator-training", name="heloc-example")
    except:
        logger.warning("Wandb not available, continuing without logging")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load HELOC data
    logger.info("Loading HELOC dataset...")
    df = pd.read_csv("heloc.csv")
    real_texts = df.apply(format_row, axis=1).tolist()
    logger.info(f"Loaded {len(real_texts)} real samples")
    
    # Initialize tokenizer and model
    logger.info("Initializing model and tokenizer...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate some synthetic data for demonstration
    logger.info("Generating synthetic data...")
    synthetic_texts = generate_synthetic_data_simple(model, tokenizer, real_texts, num_samples=50, device=device)
    logger.info(f"Generated {len(synthetic_texts)} synthetic samples")
    
    # Initialize hierarchical discriminators
    logger.info("Initializing hierarchical discriminators...")
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(
        device=device, 
        dataset_type="heloc"
    )
    
    # Initialize improved training framework
    logger.info("Initializing improved training framework...")
    improved_trainer = ImprovedDiscriminatorTraining(hierarchical_discriminators, device)
    
    # Run a small ablation study
    logger.info("Running ablation study...")
    try:
        ablation_results = improved_trainer.ablation_study(model, real_texts[:20], synthetic_texts[:20])
        logger.info("Ablation study results:")
        for key, value in ablation_results.items():
            logger.info(f"  {key}: {value:.4f}")
    except Exception as e:
        logger.warning(f"Ablation study failed: {e}")
    
    # Run grid search for weight optimization
    logger.info("Running grid search for weight optimization...")
    try:
        best_weights, best_performance, search_results = improved_trainer.grid_search_weights(
            model, real_texts[:20], synthetic_texts[:20]
        )
        logger.info(f"Best weights found: {best_weights}")
        logger.info(f"Best performance: {best_performance:.4f}")
    except Exception as e:
        logger.warning(f"Grid search failed: {e}")
    
    # Run a small alternating training cycle
    logger.info("Running alternating training...")
    try:
        training_history = improved_trainer.alternating_training(
            model, 
            real_texts[:50], 
            synthetic_texts[:50], 
            num_cycles=2,  # Small number for demonstration
            discriminator_epochs=1,
            generator_epochs=1,
            batch_size=8
        )
        logger.info(f"Completed {len(training_history)} training cycles")
    except Exception as e:
        logger.warning(f"Alternating training failed: {e}")
    
    # Generate impact analysis plot
    logger.info("Generating impact analysis plot...")
    try:
        improved_trainer.plot_impact_analysis('heloc_impact_analysis.png')
        logger.info("Impact analysis plot saved to heloc_impact_analysis.png")
    except Exception as e:
        logger.warning(f"Impact analysis plotting failed: {e}")
    
    # Save results
    logger.info("Saving results...")
    try:
        results = {
            'ablation_results': ablation_results if 'ablation_results' in locals() else None,
            'best_weights': best_weights if 'best_weights' in locals() else None,
            'best_performance': best_performance if 'best_performance' in locals() else None,
            'training_cycles': len(training_history) if 'training_history' in locals() else 0
        }
        
        import json
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to training_results.json")
    except Exception as e:
        logger.warning(f"Saving results failed: {e}")
    
    logger.info("Improved discriminator training example completed!")
    
    # Log final metrics to wandb
    if wandb.run is not None:
        if 'best_performance' in locals():
            wandb.log({"best_performance": best_performance})
        if 'training_cycles' in locals():
            wandb.log({"training_cycles": len(training_history)})
        wandb.finish()

if __name__ == "__main__":
    main()
