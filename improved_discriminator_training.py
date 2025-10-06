import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import json
import wandb
from collections import defaultdict
from typing import Dict, List, Tuple
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Bayesian optimization will be skipped.")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedDiscriminatorTraining:
    """
    Improved discriminator training with alternating updates, impact analysis, and weight tuning
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
        num_cycles=5,
        discriminator_epochs=3,
        generator_epochs=3,
        batch_size=32
    ):
        """
        Implement alternating discriminator-generator training
        """
        logger.info(f"Starting alternating training with {num_cycles} cycles")
        
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
            
            # Log cycle results
            cycle_results = {
                'cycle': cycle + 1,
                'discriminator_losses': discriminator_losses,
                'generator_losses': generator_losses
            }
            self.training_history.append(cycle_results)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    f"cycle_{cycle+1}_discriminator_loss": np.mean(discriminator_losses),
                    f"cycle_{cycle+1}_generator_loss": np.mean(generator_losses),
                    "cycle": cycle + 1
                })
        
        return self.training_history
    
    def train_discriminators_improved(
        self, 
        real_data, 
        synthetic_data, 
        epochs=3, 
        batch_size=32
    ):
        """
        Improved discriminator training with batching and monitoring
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
                
                # Train sentence discriminator
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
    
    def train_generator_with_feedback(self, generator, real_data, epochs=3, batch_size=32):
        """Train generator with discriminator feedback"""
        logger.info("Training generator with discriminator feedback...")
        
        generator.train()
        optimizer = torch.optim.AdamW(generator.parameters(), lr=5e-5)
        epoch_losses = []
        
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
                    
                    # Calculate weighted loss (using current weights)
                    token_loss = 1 - feedback["token"]
                    sentence_loss = 1 - feedback["sentence"]
                    row_loss = 1 - feedback["row"]
                    feature_loss = 1 - np.mean(list(feedback["features"].values()))
                    
                    # Store impact for analysis
                    self.impact_history['token'].append(token_loss)
                    self.impact_history['sentence'].append(sentence_loss)
                    self.impact_history['row'].append(row_loss)
                    self.impact_history['feature'].append(feature_loss)
                    
                    # Calculate total feedback loss (will be updated with tuned weights)
                    total_feedback_loss = (
                        0.2 * token_loss +
                        0.3 * sentence_loss +
                        0.3 * row_loss +
                        0.2 * feature_loss
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
                        f"generator_epoch_{epoch+1}_batch_{batch_idx}_loss": avg_batch_loss
                    })
            
            avg_epoch_loss = total_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            logger.info(f"Generator epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        return epoch_losses
    
    def ablation_study(self, generator, real_data, synthetic_data):
        """
        Perform ablation study to understand objective importance
        """
        logger.info("Performing ablation study...")
        
        base_weights = {'token': 0.2, 'sentence': 0.3, 'row': 0.3, 'feature': 0.2}
        ablation_results = {}
        
        # Test baseline performance
        baseline_performance = self.evaluate_with_weights(generator, real_data, synthetic_data, base_weights)
        ablation_results['baseline'] = baseline_performance
        
        # Test removing each objective
        for objective in ['token', 'sentence', 'row', 'feature']:
            test_weights = base_weights.copy()
            test_weights[objective] = 0.0
            
            # Normalize remaining weights
            total_weight = sum(test_weights.values())
            test_weights = {k: v/total_weight for k, v in test_weights.items()}
            
            performance = self.evaluate_with_weights(generator, real_data, synthetic_data, test_weights)
            ablation_results[f'without_{objective}'] = performance
            
            logger.info(f"Performance without {objective}: {performance:.4f}")
        
        return ablation_results
    
    def evaluate_with_weights(self, generator, real_data, synthetic_data, weights):
        """
        Evaluate generator performance with given weights
        """
        # This is a simplified evaluation - you would implement your actual evaluation metric
        total_score = 0
        
        for text in real_data[:100]:  # Sample for evaluation
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
            
            # Get feedback
            feedback = self.discriminators.get_multi_level_feedback(generated_text)
            
            # Calculate weighted score
            weighted_score = (
                weights['token'] * feedback['token'] +
                weights['sentence'] * feedback['sentence'] +
                weights['row'] * feedback['row'] +
                weights['feature'] * np.mean(list(feedback['features'].values()))
            )
            
            total_score += weighted_score
        
        return total_score / 100
    
    def grid_search_weights(self, generator, real_data, synthetic_data):
        """
        Perform grid search for optimal weight combinations
        """
        logger.info("Performing grid search for optimal weights...")
        
        weight_combinations = [
            {'token': 0.1, 'sentence': 0.3, 'row': 0.4, 'feature': 0.2},
            {'token': 0.2, 'sentence': 0.2, 'row': 0.4, 'feature': 0.2},
            {'token': 0.2, 'sentence': 0.3, 'row': 0.3, 'feature': 0.2},
            {'token': 0.2, 'sentence': 0.3, 'row': 0.2, 'feature': 0.3},
            {'token': 0.3, 'sentence': 0.2, 'row': 0.3, 'feature': 0.2},
            {'token': 0.1, 'sentence': 0.4, 'row': 0.3, 'feature': 0.2},
            {'token': 0.2, 'sentence': 0.4, 'row': 0.2, 'feature': 0.2},
            {'token': 0.3, 'sentence': 0.3, 'row': 0.2, 'feature': 0.2},
            {'token': 0.1, 'sentence': 0.2, 'row': 0.4, 'feature': 0.3},
            {'token': 0.2, 'sentence': 0.2, 'row': 0.3, 'feature': 0.3},
        ]
        
        best_weights = None
        best_performance = 0
        search_results = {}
        
        for i, weights in enumerate(weight_combinations):
            logger.info(f"Testing weight combination {i+1}/{len(weight_combinations)}: {weights}")
            
            performance = self.evaluate_with_weights(generator, real_data, synthetic_data, weights)
            search_results[f'combination_{i+1}'] = {
                'weights': weights,
                'performance': performance
            }
            
            if performance > best_performance:
                best_performance = performance
                best_weights = weights
            
            logger.info(f"Performance: {performance:.4f}")
        
        logger.info(f"Best weights found: {best_weights} with performance: {best_performance:.4f}")
        
        # Save results
        with open('grid_search_results.json', 'w') as f:
            json.dump({
                'best_weights': best_weights,
                'best_performance': best_performance,
                'all_results': search_results
            }, f, indent=2)
        
        return best_weights, best_performance, search_results
    
    def random_search_weights(self, generator, real_data, synthetic_data, n_trials=50):
        """
        Perform random search for optimal weight combinations
        """
        logger.info(f"Performing random search with {n_trials} trials...")
        
        best_weights = None
        best_performance = 0
        search_results = {}
        
        for trial in range(n_trials):
            # Generate random weights that sum to 1.0
            weights = np.random.dirichlet(np.ones(4))
            weight_dict = {
                'token': weights[0],
                'sentence': weights[1], 
                'row': weights[2],
                'feature': weights[3]
            }
            
            performance = self.evaluate_with_weights(generator, real_data, synthetic_data, weight_dict)
            search_results[f'trial_{trial+1}'] = {
                'weights': weight_dict,
                'performance': performance
            }
            
            if performance > best_performance:
                best_performance = performance
                best_weights = weight_dict
            
            if trial % 10 == 0:
                logger.info(f"Trial {trial+1}/{n_trials}, Best performance so far: {best_performance:.4f}")
        
        logger.info(f"Best weights found: {best_weights} with performance: {best_performance:.4f}")
        
        # Save results
        with open('random_search_results.json', 'w') as f:
            json.dump({
                'best_weights': best_weights,
                'best_performance': best_performance,
                'all_results': search_results
            }, f, indent=2)
        
        return best_weights, best_performance, search_results
    
    def bayesian_optimization_weights(self, generator, real_data, synthetic_data, n_trials=100):
        """
        Perform Bayesian optimization for optimal weight combinations (requires optuna)
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Skipping Bayesian optimization.")
            return None, 0, {}
        
        logger.info(f"Performing Bayesian optimization with {n_trials} trials...")
        
        def objective(trial):
            weights = {
                'token': trial.suggest_float('token', 0.0, 0.5),
                'sentence': trial.suggest_float('sentence', 0.0, 0.5),
                'row': trial.suggest_float('row', 0.0, 0.5),
                'feature': trial.suggest_float('feature', 0.0, 0.5)
            }
            
            # Ensure weights sum to 1.0
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            return self.evaluate_with_weights(generator, real_data, synthetic_data, weights)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_weights = study.best_params
        best_performance = study.best_value
        
        logger.info(f"Best weights found: {best_weights} with performance: {best_performance:.4f}")
        
        # Save results
        with open('bayesian_optimization_results.json', 'w') as f:
            json.dump({
                'best_weights': best_weights,
                'best_performance': best_performance,
                'study_history': [trial.value for trial in study.trials if trial.value is not None]
            }, f, indent=2)
        
        return best_weights, best_performance, study
    
    def plot_impact_analysis(self, save_path='impact_analysis.png'):
        """
        Plot impact analysis results
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Impact over time
        plt.subplot(2, 2, 1)
        for objective, values in self.impact_history.items():
            plt.plot(values, label=objective, alpha=0.7)
        plt.title('Objective Impact Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average impact by objective
        plt.subplot(2, 2, 2)
        avg_impacts = {obj: np.mean(values) for obj, values in self.impact_history.items()}
        plt.bar(avg_impacts.keys(), avg_impacts.values())
        plt.title('Average Impact by Objective')
        plt.ylabel('Average Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Impact distribution
        plt.subplot(2, 2, 3)
        for objective, values in self.impact_history.items():
            plt.hist(values, alpha=0.5, label=objective, bins=20)
        plt.title('Impact Distribution')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training history
        plt.subplot(2, 2, 4)
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
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Impact analysis plot saved to {save_path}")

def main():
    """
    Example usage of improved discriminator training
    """
    # Initialize wandb (optional)
    try:
        wandb.init(project="improved-discriminator-training")
    except:
        logger.warning("Wandb not available, continuing without logging")
    
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # You would initialize your hierarchical discriminators and generator here
    # hierarchical_discriminators = HierarchicalDiscriminatorSystem(device=device)
    # generator = YourGeneratorModel()
    
    # Load data
    # df = pd.read_csv("heloc.csv")
    # real_texts = df.apply(format_row, axis=1).tolist()
    # synthetic_texts = generate_synthetic_data(generator, real_texts)
    
    # Initialize improved training
    # improved_trainer = ImprovedDiscriminatorTraining(hierarchical_discriminators, device)
    
    # Run alternating training
    # training_history = improved_trainer.alternating_training(
    #     generator, real_texts, synthetic_texts, num_cycles=5
    # )
    
    # Run ablation study
    # ablation_results = improved_trainer.ablation_study(generator, real_texts, synthetic_texts)
    
    # Run weight tuning
    # best_weights, best_performance, search_results = improved_trainer.grid_search_weights(
    #     generator, real_texts, synthetic_texts
    # )
    
    # Plot impact analysis
    # improved_trainer.plot_impact_analysis()
    
    logger.info("Improved discriminator training framework ready!")

if __name__ == "__main__":
    main()
