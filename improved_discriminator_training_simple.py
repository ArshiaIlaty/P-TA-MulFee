import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import json
import wandb
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
from utils import format_row
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedDiscriminatorTrainingSimple:
    """
    Simplified improved discriminator training with core fixes
    """
    
    def __init__(self, hierarchical_discriminators, device="cuda"):
        self.discriminators = hierarchical_discriminators
        self.device = device
        self.impact_history = defaultdict(list)
        self.training_history = []
        
    def alternating_training(
        self, 
        generator, 
        real_data, 
        synthetic_data, 
        num_cycles=5,  # Increased from 2 to 5
        discriminator_epochs=2,
        generator_epochs=2,
        batch_size=16,  # Increased from 8 to 16
        early_stopping_patience=3
    ):
        """
        Implement alternating discriminator-generator training with early stopping
        """
        logger.info(f"Starting alternating training with {num_cycles} cycles")
        
        best_generator_loss = float('inf')
        patience_counter = 0
        
        for cycle in range(num_cycles):
            logger.info(f"=== CYCLE {cycle + 1}/{num_cycles} ===")
            
            # Step 1: Train discriminators
            logger.info("Training discriminators...")
            discriminator_losses = self.train_discriminators_improved(
                real_data, synthetic_data, epochs=discriminator_epochs, batch_size=batch_size
            )
            
            # Step 2: Train generator with discriminator feedback
            logger.info("Training generator with discriminator feedback...")
            generator_losses = self.train_generator_with_feedback(
                generator, real_data, epochs=generator_epochs, batch_size=batch_size
            )
            
            # Early stopping check
            current_generator_loss = np.mean(generator_losses)
            if current_generator_loss < best_generator_loss:
                best_generator_loss = current_generator_loss
                patience_counter = 0
                logger.info(f"New best generator loss: {best_generator_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Generator loss not improving. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at cycle {cycle + 1}")
                break
            
            # Log cycle results
            cycle_results = {
                'cycle': cycle + 1,
                'discriminator_losses': discriminator_losses,
                'generator_losses': generator_losses,
                'best_generator_loss': best_generator_loss
            }
            self.training_history.append(cycle_results)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    f"cycle_{cycle+1}_discriminator_loss": np.mean(discriminator_losses),
                    f"cycle_{cycle+1}_generator_loss": np.mean(generator_losses),
                    f"cycle_{cycle+1}_best_generator_loss": best_generator_loss,
                    "cycle": cycle + 1
                })
        
        return self.training_history
    
    def train_discriminators_improved(
        self, 
        real_data, 
        synthetic_data, 
        epochs=2, 
        batch_size=16
    ):
        """
        Improved discriminator training with better batching
        """
        logger.info("Training discriminators with improved batching...")
        
        # Prepare data with batching
        all_texts = real_data + synthetic_data
        real_labels = torch.ones(len(real_data)).to(self.device)
        synthetic_labels = torch.zeros(len(synthetic_data)).to(self.device)
        all_labels = torch.cat([real_labels, synthetic_labels])
        
        # Create batches
        num_batches = len(all_texts) // batch_size
        batch_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_texts = all_texts[start_idx:end_idx]
                batch_labels = all_labels[start_idx:end_idx]
                
                # Train token discriminator
                token_loss = self.train_token_discriminator_batch(batch_texts, batch_labels)
                
                # Train sentence discriminator (keep original, but monitor closely)
                sentence_loss = self.train_sentence_discriminator_batch(batch_texts, batch_labels)
                
                # Train row discriminator
                row_loss = self.train_row_discriminator_batch(batch_texts, batch_labels)
                
                # Train feature discriminators
                feature_loss = self.train_feature_discriminators_batch(batch_texts, batch_labels)
                
                # Calculate total batch loss
                total_batch_loss = token_loss + sentence_loss + row_loss + feature_loss
                epoch_losses.append(total_batch_loss)
                
                # Log individual losses
                if wandb.run is not None:
                    wandb.log({
                        f"epoch_{epoch+1}_batch_{batch_idx}_token_loss": token_loss,
                        f"epoch_{epoch+1}_batch_{batch_idx}_sentence_loss": sentence_loss,
                        f"epoch_{epoch+1}_batch_{batch_idx}_row_loss": row_loss,
                        f"epoch_{epoch+1}_batch_{batch_idx}_feature_loss": feature_loss,
                        f"epoch_{epoch+1}_batch_{batch_idx}_total_loss": total_batch_loss
                    })
                
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {total_batch_loss:.4f}")
            
            avg_epoch_loss = np.mean(epoch_losses)
            batch_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        return batch_losses
    
    def train_token_discriminator_batch(self, texts, labels):
        """Train token discriminator on a batch"""
        total_loss = 0
        
        for text, label in zip(texts, labels):
            token_ids = self.discriminators.tokenizer.encode(text, return_tensors="pt").to(self.device)
            label = label.unsqueeze(0).float()
            
            self.discriminators.optimizers["token"].zero_grad()
            prediction = self.discriminators.token_discriminator(token_ids)
            
            if prediction.shape != label.shape:
                prediction = prediction.squeeze()
                if prediction.dim() == 0:
                    prediction = prediction.unsqueeze(0)
            
            loss = F.binary_cross_entropy(prediction, label)
            loss.backward()
            self.discriminators.optimizers["token"].step()
            total_loss += loss.item()
        
        return total_loss / len(texts)
    
    def train_sentence_discriminator_batch(self, texts, labels):
        """Train sentence discriminator on a batch"""
        total_loss = 0
        num_sentences = 0
        
        for text, label in zip(texts, labels):
            sentences = text.split(", ")
            for sentence in sentences:
                if " is " in sentence:
                    try:
                        encoding = self.discriminators.tokenizer(
                            sentence,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=64,
                        ).to(self.device)
                        label_tensor = label.unsqueeze(0).float()
                        
                        self.discriminators.optimizers["sentence"].zero_grad()
                        prediction = self.discriminators.sentence_discriminator(
                            encoding["input_ids"], encoding["attention_mask"]
                        )
                        
                        if prediction.shape != label_tensor.shape:
                            prediction = prediction.squeeze()
                            if prediction.dim() == 0:
                                prediction = prediction.unsqueeze(0)
                        
                        loss = F.binary_cross_entropy(prediction, label_tensor)
                        loss.backward()
                        self.discriminators.optimizers["sentence"].step()
                        total_loss += loss.item()
                        num_sentences += 1
                        
                    except Exception as e:
                        logger.warning(f"Sentence training error: {e}")
                        continue
        
        return total_loss / max(num_sentences, 1)
    
    def train_row_discriminator_batch(self, texts, labels):
        """Train row discriminator on a batch"""
        total_loss = 0
        
        for text, label in zip(texts, labels):
            encoding = self.discriminators.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            label = label.unsqueeze(0).float()
            
            self.discriminators.optimizers["row"].zero_grad()
            prediction = self.discriminators.row_discriminator(
                encoding["input_ids"], encoding["attention_mask"]
            )
            
            if prediction.shape != label.shape:
                prediction = prediction.squeeze()
                if prediction.dim() == 0:
                    prediction = prediction.unsqueeze(0)
            
            loss = F.binary_cross_entropy(prediction, label)
            loss.backward()
            self.discriminators.optimizers["row"].step()
            total_loss += loss.item()
        
        return total_loss / len(texts)
    
    def train_feature_discriminators_batch(self, texts, labels):
        """Train feature discriminators on a batch"""
        total_loss = 0
        num_features = 0
        
        for text, label in zip(texts, labels):
            for feature_name, discriminator in self.discriminators.feature_discriminators.items():
                embeddings = self.discriminators.extract_feature_embeddings(text, feature_name)
                label_tensor = label.unsqueeze(0).float()
                
                self.discriminators.optimizers["features"][feature_name].zero_grad()
                prediction = discriminator(embeddings)
                
                if prediction.shape != label_tensor.shape:
                    prediction = prediction.squeeze()
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)
                
                loss = F.binary_cross_entropy(prediction, label_tensor)
                loss.backward()
                self.discriminators.optimizers["features"][feature_name].step()
                total_loss += loss.item()
                num_features += 1
        
        return total_loss / max(num_features, 1)
    
    def train_generator_with_feedback(self, generator, real_data, epochs=2, batch_size=16):
        """Train generator with optimized discriminator feedback weights"""
        logger.info("Training generator with optimized feedback weights...")
        
        generator.train()
        optimizer = torch.optim.AdamW(generator.parameters(), lr=5e-5)
        epoch_losses = []
        
        # Optimized weights based on analysis - disable sentence discriminator
        optimized_weights = {
            'token': 0.2,
            'sentence': 0.0,  # Disabled due to poor performance
            'row': 0.4,       # Increased importance
            'feature': 0.4    # Increased importance
        }
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = len(real_data) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_texts = real_data[start_idx:end_idx]
                
                batch_loss = 0
                for text in batch_texts:
                    # Generate synthetic text
                    input_ids = self.discriminators.tokenizer.encode(text, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = generator.generate(
                            input_ids,
                            max_length=input_ids.shape[1] + 20,
                            pad_token_id=self.discriminators.tokenizer.eos_token_id,
                            do_sample=True,
                            top_p=0.9,
                        )
                    
                    generated_text = self.discriminators.tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )
                    
                    # Get multi-level feedback
                    feedback = self.discriminators.get_multi_level_feedback(generated_text)
                    
                    # Calculate weighted loss with optimized weights
                    token_loss = 1 - feedback["token"]
                    sentence_loss = 1 - feedback["sentence"]
                    row_loss = 1 - feedback["row"]
                    feature_loss = 1 - np.mean(list(feedback["features"].values()))
                    
                    # Store impact for analysis
                    self.impact_history['token'].append(token_loss)
                    self.impact_history['sentence'].append(sentence_loss)
                    self.impact_history['row'].append(row_loss)
                    self.impact_history['feature'].append(feature_loss)
                    
                    # Calculate total feedback loss with optimized weights
                    total_feedback_loss = (
                        optimized_weights['token'] * token_loss +
                        optimized_weights['sentence'] * sentence_loss +
                        optimized_weights['row'] * row_loss +
                        optimized_weights['feature'] * feature_loss
                    )
                    
                    # Standard language modeling loss
                    outputs = generator(input_ids=input_ids, labels=input_ids)
                    lm_loss = outputs.loss
                    
                    # Combined loss
                    combined_loss = lm_loss + 0.1 * total_feedback_loss
                    batch_loss += combined_loss.item()
                
                # Update generator
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()
                
                avg_batch_loss = batch_loss / len(batch_texts)
                total_loss += avg_batch_loss
                
                if wandb.run is not None:
                    wandb.log({
                        f"generator_epoch_{epoch+1}_batch_{batch_idx}_loss": avg_batch_loss,
                        f"generator_epoch_{epoch+1}_batch_{batch_idx}_token_loss": token_loss,
                        f"generator_epoch_{epoch+1}_batch_{batch_idx}_sentence_loss": sentence_loss,
                        f"generator_epoch_{epoch+1}_batch_{batch_idx}_row_loss": row_loss,
                        f"generator_epoch_{epoch+1}_batch_{batch_idx}_feature_loss": feature_loss
                    })
            
            avg_epoch_loss = total_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            logger.info(f"Generator epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        return epoch_losses
    
    def evaluate_discriminator_performance(self, real_data, synthetic_data, num_samples=50):
        """
        Evaluate discriminator performance with separation metrics
        """
        logger.info("Evaluating discriminator performance...")
        
        # Sample data for evaluation
        real_sample = real_data[:num_samples//2]
        synthetic_sample = synthetic_data[:num_samples//2]
        
        results = {
            'token': {'real_scores': [], 'synthetic_scores': []},
            'sentence': {'real_scores': [], 'synthetic_scores': []},
            'row': {'real_scores': [], 'synthetic_scores': []},
            'feature': {'real_scores': [], 'synthetic_scores': []}
        }
        
        # Evaluate real data
        for text in real_sample:
            try:
                feedback = self.discriminators.get_multi_level_feedback(text)
                results['token']['real_scores'].append(feedback['token'])
                results['sentence']['real_scores'].append(feedback['sentence'])
                results['row']['real_scores'].append(feedback['row'])
                results['feature']['real_scores'].append(np.mean(list(feedback['features'].values())))
            except Exception as e:
                logger.warning(f"Error evaluating real text: {e}")
                continue
        
        # Evaluate synthetic data
        for text in synthetic_sample:
            try:
                feedback = self.discriminators.get_multi_level_feedback(text)
                results['token']['synthetic_scores'].append(feedback['token'])
                results['sentence']['synthetic_scores'].append(feedback['sentence'])
                results['row']['synthetic_scores'].append(feedback['row'])
                results['feature']['synthetic_scores'].append(np.mean(list(feedback['features'].values())))
            except Exception as e:
                logger.warning(f"Error evaluating synthetic text: {e}")
                continue
        
        # Calculate metrics
        metrics = {}
        for discriminator_type, scores in results.items():
            if len(scores['real_scores']) > 0 and len(scores['synthetic_scores']) > 0:
                real_scores = np.array(scores['real_scores'])
                synthetic_scores = np.array(scores['synthetic_scores'])
                
                # Calculate separation (higher is better)
                separation = np.mean(real_scores) - np.mean(synthetic_scores)
                
                # Calculate accuracy (assuming threshold at 0.5)
                real_correct = np.sum(real_scores > 0.5)
                synthetic_correct = np.sum(synthetic_scores < 0.5)
                accuracy = (real_correct + synthetic_correct) / (len(real_scores) + len(synthetic_scores))
                
                metrics[discriminator_type] = {
                    'separation': separation,
                    'accuracy': accuracy,
                    'real_mean': np.mean(real_scores),
                    'synthetic_mean': np.mean(synthetic_scores)
                }
            else:
                metrics[discriminator_type] = {
                    'separation': 0.0,
                    'accuracy': 0.0,
                    'real_mean': 0.0,
                    'synthetic_mean': 0.0
                }
        
        return metrics, results
    
    def plot_impact_analysis_simple(self, save_path='impact_analysis_simple.png'):
        """
        Simple impact analysis plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Impact over time
        plt.subplot(2, 3, 1)
        for objective, values in self.impact_history.items():
            if values:  # Only plot if there are values
                plt.plot(values, label=objective, alpha=0.7)
        plt.title('Objective Impact Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average impact by objective
        plt.subplot(2, 3, 2)
        avg_impacts = {obj: np.mean(values) for obj, values in self.impact_history.items() if values}
        if avg_impacts:
            plt.bar(avg_impacts.keys(), avg_impacts.values())
        plt.title('Average Impact by Objective')
        plt.ylabel('Average Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Impact distribution
        plt.subplot(2, 3, 3)
        for objective, values in self.impact_history.items():
            if values:  # Only plot if there are values
                plt.hist(values, alpha=0.5, label=objective, bins=20)
        plt.title('Impact Distribution')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training history
        plt.subplot(2, 3, 4)
        if self.training_history:
            cycles = [h['cycle'] for h in self.training_history]
            disc_losses = [np.mean(h['discriminator_losses']) for h in self.training_history]
            gen_losses = [np.mean(h['generator_losses']) for h in self.training_history]
            
            plt.plot(cycles, disc_losses, 'o-', label='Discriminator Loss')
            plt.plot(cycles, gen_losses, 's-', label='Generator Loss')
            plt.title('Training History')
            plt.xlabel('Cycle')
            plt.ylabel('Average Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Loss components breakdown
        plt.subplot(2, 3, 5)
        if self.training_history and any(self.impact_history.values()):
            components = ['token', 'sentence', 'row', 'feature']
            avg_losses = [
                np.mean(self.impact_history['token'][-10:]) if self.impact_history['token'] else 0,
                np.mean(self.impact_history['sentence'][-10:]) if self.impact_history['sentence'] else 0,
                np.mean(self.impact_history['row'][-10:]) if self.impact_history['row'] else 0,
                np.mean(self.impact_history['feature'][-10:]) if self.impact_history['feature'] else 0
            ]
            plt.bar(components, avg_losses)
            plt.title('Recent Loss Components')
            plt.ylabel('Average Loss')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Training progress
        plt.subplot(2, 3, 6)
        if self.training_history:
            cycles = [h['cycle'] for h in self.training_history]
            best_losses = [h['best_generator_loss'] for h in self.training_history]
            plt.plot(cycles, best_losses, 'o-', color='red', linewidth=2)
            plt.title('Best Generator Loss Progress')
            plt.xlabel('Cycle')
            plt.ylabel('Best Loss')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Simple impact analysis plot saved to {save_path}")

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
    """
    Example usage of simplified improved discriminator training
    """
    # Initialize wandb
    try:
        wandb.init(project="improved-discriminator-training", name="heloc-simple-fixed")
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
    synthetic_texts = generate_synthetic_data_simple(model, tokenizer, real_texts, num_samples=100, device=device)
    logger.info(f"Generated {len(synthetic_texts)} synthetic samples")
    
    # Initialize hierarchical discriminators
    logger.info("Initializing hierarchical discriminators...")
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(
        device=device, 
        dataset_type="heloc"
    )
    
    # Initialize improved training framework
    logger.info("Initializing improved training framework...")
    improved_trainer = ImprovedDiscriminatorTrainingSimple(hierarchical_discriminators, device)
    
    # Evaluate initial discriminator performance
    logger.info("Evaluating initial discriminator performance...")
    initial_metrics, initial_results = improved_trainer.evaluate_discriminator_performance(
        real_texts[:50], synthetic_texts[:50]
    )
    logger.info("Initial discriminator metrics:")
    for discriminator, metrics in initial_metrics.items():
        logger.info(f"  {discriminator}: separation={metrics['separation']:.4f}, accuracy={metrics['accuracy']:.4f}")
    
    # Run improved alternating training
    logger.info("Running improved alternating training...")
    training_history = improved_trainer.alternating_training(
        model, 
        real_texts[:100], 
        synthetic_texts[:100], 
        num_cycles=5,  # Increased cycles
        discriminator_epochs=2,
        generator_epochs=2,
        batch_size=16,
        early_stopping_patience=3
    )
    logger.info(f"Completed {len(training_history)} training cycles")
    
    # Evaluate final discriminator performance
    logger.info("Evaluating final discriminator performance...")
    final_metrics, final_results = improved_trainer.evaluate_discriminator_performance(
        real_texts[:50], synthetic_texts[:50]
    )
    logger.info("Final discriminator metrics:")
    for discriminator, metrics in final_metrics.items():
        logger.info(f"  {discriminator}: separation={metrics['separation']:.4f}, accuracy={metrics['accuracy']:.4f}")
    
    # Generate impact analysis plot
    logger.info("Generating impact analysis plot...")
    improved_trainer.plot_impact_analysis_simple('heloc_impact_analysis_simple.png')
    logger.info("Impact analysis plot saved to heloc_impact_analysis_simple.png")
    
    # Save comprehensive results
    logger.info("Saving comprehensive results...")
    results = {
        'training_cycles': len(training_history),
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'best_generator_loss': training_history[-1]['best_generator_loss'] if training_history else None,
        'training_history': training_history
    }
    
    with open('training_results_simple.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Comprehensive results saved to training_results_simple.json")
    
    logger.info("Simplified improved discriminator training completed!")
    
    # Log final metrics to wandb
    if wandb.run is not None:
        if training_history:
            wandb.log({
                "final_best_generator_loss": training_history[-1]['best_generator_loss'],
                "training_cycles": len(training_history)
            })
        wandb.finish()

if __name__ == "__main__":
    main()
