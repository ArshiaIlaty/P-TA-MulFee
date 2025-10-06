import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import json
import wandb
import os
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
from utils import format_row
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedDiscriminatorTrainingRobust:
    """
    Robust improved discriminator training with graceful error handling
    """
    
    def __init__(self, hierarchical_discriminators, device="cuda", checkpoint_dir="checkpoints"):
        self.discriminators = hierarchical_discriminators
        self.device = device
        self.impact_history = defaultdict(list)
        self.training_history = []
        self.checkpoint_dir = checkpoint_dir
        self.bert_fix_successful = False
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Try to fix sentence discriminator with BERT, but continue if it fails
        self.try_fix_sentence_discriminator()
        
    def try_fix_sentence_discriminator(self):
        """
        Try to replace GPT-2 sentence discriminator with BERT, but continue if it fails
        """
        logger.info("Attempting to fix sentence discriminator with BERT...")
        
        try:
            # Use DistilBERT for efficiency
            sentence_model_name = "distilbert-base-uncased"
            sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_model_name)
            sentence_model = AutoModelForSequenceClassification.from_pretrained(
                sentence_model_name, num_labels=1
            ).to(self.device)
            
            # Test the model with a simple input to catch CUDA errors early
            test_input = sentence_tokenizer(
                "test sentence",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(self.device)
            
            with torch.no_grad():
                test_output = sentence_model(**test_input)
            
            # If we get here, the model works
            self.discriminators.sentence_discriminator = sentence_model
            self.discriminators.sentence_tokenizer = sentence_tokenizer
            
            # Update optimizer
            self.discriminators.optimizers["sentence"] = torch.optim.AdamW(
                sentence_model.parameters(), lr=1e-4
            )
            
            self.bert_fix_successful = True
            logger.info("✅ Sentence discriminator successfully replaced with DistilBERT")
            
        except Exception as e:
            logger.warning(f"❌ Failed to replace sentence discriminator with BERT: {e}")
            logger.info("🔄 Continuing with original sentence discriminator")
            self.bert_fix_successful = False
    
    def save_checkpoint(self, generator, cycle, best_generator_loss, training_history):
        """
        Save training checkpoint
        """
        checkpoint = {
            'cycle': cycle,
            'best_generator_loss': best_generator_loss,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': {
                'token': self.discriminators.token_discriminator.state_dict(),
                'sentence': self.discriminators.sentence_discriminator.state_dict(),
                'row': self.discriminators.row_discriminator.state_dict(),
                'feature': {name: disc.state_dict() for name, disc in self.discriminators.feature_discriminators.items()}
            },
            'optimizer_state_dict': {
                'token': self.discriminators.optimizers["token"].state_dict(),
                'sentence': self.discriminators.optimizers["sentence"].state_dict(),
                'row': self.discriminators.optimizers["row"].state_dict(),
                'feature': {name: opt.state_dict() for name, opt in self.discriminators.optimizers["features"].items()}
            },
            'training_history': training_history,
            'impact_history': dict(self.impact_history),
            'bert_fix_successful': self.bert_fix_successful
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_cycle_{cycle}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved to {checkpoint_path}")
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, generator, checkpoint_path):
        """
        Load training checkpoint
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load generator state
            generator.load_state_dict(checkpoint['generator_state_dict'])
            
            # Load discriminator states
            self.discriminators.token_discriminator.load_state_dict(checkpoint['discriminator_state_dict']['token'])
            self.discriminators.sentence_discriminator.load_state_dict(checkpoint['discriminator_state_dict']['sentence'])
            self.discriminators.row_discriminator.load_state_dict(checkpoint['discriminator_state_dict']['row'])
            
            for name, state_dict in checkpoint['discriminator_state_dict']['feature'].items():
                if name in self.discriminators.feature_discriminators:
                    self.discriminators.feature_discriminators[name].load_state_dict(state_dict)
            
            # Load optimizer states
            self.discriminators.optimizers["token"].load_state_dict(checkpoint['optimizer_state_dict']['token'])
            self.discriminators.optimizers["sentence"].load_state_dict(checkpoint['optimizer_state_dict']['sentence'])
            self.discriminators.optimizers["row"].load_state_dict(checkpoint['optimizer_state_dict']['row'])
            
            for name, state_dict in checkpoint['optimizer_state_dict']['feature'].items():
                if name in self.discriminators.optimizers["features"]:
                    self.discriminators.optimizers["features"][name].load_state_dict(state_dict)
            
            # Load training history
            self.training_history = checkpoint['training_history']
            self.impact_history = defaultdict(list, checkpoint['impact_history'])
            self.bert_fix_successful = checkpoint.get('bert_fix_successful', False)
            
            logger.info(f"📂 Checkpoint loaded from {checkpoint_path}")
            logger.info(f"🔄 Resuming from cycle {checkpoint['cycle']} with best loss {checkpoint['best_generator_loss']:.4f}")
            logger.info(f"🤖 BERT fix status: {'✅ Successful' if self.bert_fix_successful else '❌ Failed'}")
            
            return checkpoint['cycle'], checkpoint['best_generator_loss']
        
        return 0, float('inf')
    
    def alternating_training(
        self, 
        generator, 
        real_data, 
        synthetic_data, 
        num_cycles=10,
        discriminator_epochs=2,
        generator_epochs=2,
        batch_size=16,
        early_stopping_patience=5,
        resume_from_checkpoint=True
    ):
        """
        Implement alternating discriminator-generator training with checkpointing
        """
        logger.info(f"🚀 Starting alternating training with {num_cycles} cycles")
        logger.info(f"🤖 BERT fix status: {'✅ Successful' if self.bert_fix_successful else '❌ Failed'}")
        
        # Try to resume from checkpoint
        start_cycle = 0
        best_generator_loss = float('inf')
        
        if resume_from_checkpoint:
            latest_checkpoint = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_checkpoint):
                start_cycle, best_generator_loss = self.load_checkpoint(generator, latest_checkpoint)
                start_cycle += 1  # Start from next cycle
        
        patience_counter = 0
        
        for cycle in range(start_cycle, num_cycles):
            logger.info(f"🔄 === CYCLE {cycle + 1}/{num_cycles} ===")
            
            # Step 1: Train discriminators
            logger.info("🎯 Training discriminators...")
            discriminator_losses = self.train_discriminators_improved(
                real_data, synthetic_data, epochs=discriminator_epochs, batch_size=batch_size
            )
            
            # Step 2: Train generator with discriminator feedback
            logger.info("🤖 Training generator with discriminator feedback...")
            generator_losses = self.train_generator_with_feedback(
                generator, real_data, epochs=generator_epochs, batch_size=batch_size
            )
            
            # Early stopping check
            current_generator_loss = np.mean(generator_losses)
            if current_generator_loss < best_generator_loss:
                best_generator_loss = current_generator_loss
                patience_counter = 0
                logger.info(f"🏆 New best generator loss: {best_generator_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"⏳ Generator loss not improving. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"🛑 Early stopping triggered at cycle {cycle + 1}")
                break
            
            # Log cycle results
            cycle_results = {
                'cycle': cycle + 1,
                'discriminator_losses': discriminator_losses,
                'generator_losses': generator_losses,
                'best_generator_loss': best_generator_loss
            }
            self.training_history.append(cycle_results)
            
            # Save checkpoint every 2 cycles
            if (cycle + 1) % 2 == 0:
                self.save_checkpoint(generator, cycle + 1, best_generator_loss, self.training_history)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    f"cycle_{cycle+1}_discriminator_loss": np.mean(discriminator_losses),
                    f"cycle_{cycle+1}_generator_loss": np.mean(generator_losses),
                    f"cycle_{cycle+1}_best_generator_loss": best_generator_loss,
                    "cycle": cycle + 1,
                    "bert_fix_successful": self.bert_fix_successful
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
        Improved discriminator training with robust sentence discriminator
        """
        logger.info("🎯 Training discriminators with improved batching...")
        
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
                
                # Train sentence discriminator (with appropriate method)
                if self.bert_fix_successful:
                    sentence_loss = self.train_sentence_discriminator_bert(batch_texts, batch_labels)
                else:
                    sentence_loss = self.train_sentence_discriminator_original(batch_texts, batch_labels)
                
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
                
                logger.info(f"📊 Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {total_batch_loss:.4f}")
            
            avg_epoch_loss = np.mean(epoch_losses)
            batch_losses.append(avg_epoch_loss)
            logger.info(f"📈 Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        return batch_losses
    
    def train_sentence_discriminator_bert(self, texts, labels):
        """Train sentence discriminator with BERT on a batch"""
        total_loss = 0
        num_sentences = 0
        
        for text, label in zip(texts, labels):
            sentences = text.split(", ")
            for sentence in sentences:
                if " is " in sentence:
                    try:
                        # Use BERT tokenizer
                        encoding = self.discriminators.sentence_tokenizer(
                            sentence,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=64,
                        ).to(self.device)
                        label_tensor = label.unsqueeze(0).float()
                        
                        self.discriminators.optimizers["sentence"].zero_grad()
                        prediction = self.discriminators.sentence_discriminator(**encoding).logits
                        
                        if prediction.shape != label_tensor.shape:
                            prediction = prediction.squeeze()
                            if prediction.dim() == 0:
                                prediction = prediction.unsqueeze(0)
                        
                        loss = F.binary_cross_entropy_with_logits(prediction, label_tensor)
                        loss.backward()
                        self.discriminators.optimizers["sentence"].step()
                        total_loss += loss.item()
                        num_sentences += 1
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Sentence BERT training error: {e}")
                        continue
        
        return total_loss / max(num_sentences, 1)
    
    def train_sentence_discriminator_original(self, texts, labels):
        """Train sentence discriminator with original method on a batch"""
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
                        logger.warning(f"⚠️ Sentence original training error: {e}")
                        continue
        
        return total_loss / max(num_sentences, 1)
    
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
        """Train generator with balanced discriminator feedback weights"""
        logger.info("🤖 Training generator with balanced feedback weights...")
        
        generator.train()
        optimizer = torch.optim.AdamW(generator.parameters(), lr=5e-5)
        epoch_losses = []
        
        # Balanced weights - adjust based on BERT fix status
        if self.bert_fix_successful:
            balanced_weights = {
                'token': 0.2,
                'sentence': 0.2,  # Enable with BERT fix
                'row': 0.3,       # Good performer
                'feature': 0.3    # Good performer
            }
        else:
            balanced_weights = {
                'token': 0.2,
                'sentence': 0.0,  # Disable without BERT fix
                'row': 0.4,       # Increase weight
                'feature': 0.4    # Increase weight
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
                    
                    # Calculate weighted loss with balanced weights
                    token_loss = 1 - feedback["token"]
                    sentence_loss = 1 - feedback["sentence"]
                    row_loss = 1 - feedback["row"]
                    feature_loss = 1 - np.mean(list(feedback["features"].values()))
                    
                    # Store impact for analysis
                    self.impact_history['token'].append(token_loss)
                    self.impact_history['sentence'].append(sentence_loss)
                    self.impact_history['row'].append(row_loss)
                    self.impact_history['feature'].append(feature_loss)
                    
                    # Calculate total feedback loss with balanced weights
                    total_feedback_loss = (
                        balanced_weights['token'] * token_loss +
                        balanced_weights['sentence'] * sentence_loss +
                        balanced_weights['row'] * row_loss +
                        balanced_weights['feature'] * feature_loss
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
            logger.info(f"📈 Generator epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        return epoch_losses
    
    def evaluate_discriminator_performance(self, real_data, synthetic_data, num_samples=50):
        """
        Evaluate discriminator performance with separation metrics
        """
        logger.info("📊 Evaluating discriminator performance...")
        
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
                logger.warning(f"⚠️ Error evaluating real text: {e}")
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
                logger.warning(f"⚠️ Error evaluating synthetic text: {e}")
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
    
    def plot_impact_analysis_robust(self, save_path='impact_analysis_robust.png'):
        """
        Robust impact analysis plot with comprehensive metrics
        """
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Impact over time
        plt.subplot(3, 3, 1)
        for objective, values in self.impact_history.items():
            if values:  # Only plot if there are values
                plt.plot(values, label=objective, alpha=0.7)
        plt.title('Objective Impact Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average impact by objective
        plt.subplot(3, 3, 2)
        avg_impacts = {obj: np.mean(values) for obj, values in self.impact_history.items() if values}
        if avg_impacts:
            plt.bar(avg_impacts.keys(), avg_impacts.values())
        plt.title('Average Impact by Objective')
        plt.ylabel('Average Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Impact distribution
        plt.subplot(3, 3, 3)
        for objective, values in self.impact_history.items():
            if values:  # Only plot if there are values
                plt.hist(values, alpha=0.5, label=objective, bins=20)
        plt.title('Impact Distribution')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training history
        plt.subplot(3, 3, 4)
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
        plt.subplot(3, 3, 5)
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
        plt.subplot(3, 3, 6)
        if self.training_history:
            cycles = [h['cycle'] for h in self.training_history]
            best_losses = [h['best_generator_loss'] for h in self.training_history]
            plt.plot(cycles, best_losses, 'o-', color='red', linewidth=2)
            plt.title('Best Generator Loss Progress')
            plt.xlabel('Cycle')
            plt.ylabel('Best Loss')
            plt.grid(True, alpha=0.3)
        
        # Plot 7: Loss convergence
        plt.subplot(3, 3, 7)
        if self.training_history:
            cycles = [h['cycle'] for h in self.training_history]
            disc_losses = [np.mean(h['discriminator_losses']) for h in self.training_history]
            gen_losses = [np.mean(h['generator_losses']) for h in self.training_history]
            
            plt.semilogy(cycles, disc_losses, 'o-', label='Discriminator Loss')
            plt.semilogy(cycles, gen_losses, 's-', label='Generator Loss')
            plt.title('Loss Convergence (Log Scale)')
            plt.xlabel('Cycle')
            plt.ylabel('Average Loss (Log)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 8: Improvement rate
        plt.subplot(3, 3, 8)
        if len(self.training_history) > 1:
            cycles = [h['cycle'] for h in self.training_history]
            gen_losses = [np.mean(h['generator_losses']) for h in self.training_history]
            improvements = [0] + [gen_losses[i-1] - gen_losses[i] for i in range(1, len(gen_losses))]
            
            plt.bar(cycles, improvements)
            plt.title('Generator Loss Improvement per Cycle')
            plt.xlabel('Cycle')
            plt.ylabel('Loss Improvement')
            plt.grid(True, alpha=0.3)
        
        # Plot 9: Training efficiency
        plt.subplot(3, 3, 9)
        if self.training_history:
            cycles = [h['cycle'] for h in self.training_history]
            disc_losses = [np.mean(h['discriminator_losses']) for h in self.training_history]
            gen_losses = [np.mean(h['generator_losses']) for h in self.training_history]
            
            # Calculate efficiency ratio
            efficiency = [g/d if d > 0 else 0 for g, d in zip(gen_losses, disc_losses)]
            plt.plot(cycles, efficiency, 'o-', color='green', linewidth=2)
            plt.title('Training Efficiency (Gen/Disc Loss Ratio)')
            plt.xlabel('Cycle')
            plt.ylabel('Efficiency Ratio')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Robust impact analysis plot saved to {save_path}")

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
    Example usage of robust improved discriminator training
    """
    # Initialize wandb
    try:
        wandb.init(project="improved-discriminator-training", name="heloc-robust-fixed")
    except:
        logger.warning("⚠️ Wandb not available, continuing without logging")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ Using device: {device}")
    
    # Load HELOC data
    logger.info("📂 Loading HELOC dataset...")
    df = pd.read_csv("heloc.csv")
    real_texts = df.apply(format_row, axis=1).tolist()
    logger.info(f"✅ Loaded {len(real_texts)} real samples")
    
    # Initialize tokenizer and model
    logger.info("🤖 Initializing model and tokenizer...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate some synthetic data for demonstration
    logger.info("🔄 Generating synthetic data...")
    synthetic_texts = generate_synthetic_data_simple(model, tokenizer, real_texts, num_samples=100, device=device)
    logger.info(f"✅ Generated {len(synthetic_texts)} synthetic samples")
    
    # Initialize hierarchical discriminators
    logger.info("🎯 Initializing hierarchical discriminators...")
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(
        device=device, 
        dataset_type="heloc"
    )
    
    # Initialize improved training framework
    logger.info("🚀 Initializing improved training framework...")
    improved_trainer = ImprovedDiscriminatorTrainingRobust(hierarchical_discriminators, device)
    
    # Evaluate initial discriminator performance
    logger.info("📊 Evaluating initial discriminator performance...")
    initial_metrics, initial_results = improved_trainer.evaluate_discriminator_performance(
        real_texts[:50], synthetic_texts[:50]
    )
    logger.info("📈 Initial discriminator metrics:")
    for discriminator, metrics in initial_metrics.items():
        logger.info(f"  {discriminator}: separation={metrics['separation']:.4f}, accuracy={metrics['accuracy']:.4f}")
    
    # Run improved alternating training
    logger.info("🔄 Running improved alternating training...")
    training_history = improved_trainer.alternating_training(
        model, 
        real_texts[:100], 
        synthetic_texts[:100], 
        num_cycles=10,
        discriminator_epochs=2,
        generator_epochs=2,
        batch_size=16,
        early_stopping_patience=5,
        resume_from_checkpoint=True
    )
    logger.info(f"✅ Completed {len(training_history)} training cycles")
    
    # Evaluate final discriminator performance
    logger.info("📊 Evaluating final discriminator performance...")
    final_metrics, final_results = improved_trainer.evaluate_discriminator_performance(
        real_texts[:50], synthetic_texts[:50]
    )
    logger.info("📈 Final discriminator metrics:")
    for discriminator, metrics in final_metrics.items():
        logger.info(f"  {discriminator}: separation={metrics['separation']:.4f}, accuracy={metrics['accuracy']:.4f}")
    
    # Generate robust impact analysis plot
    logger.info("📊 Generating robust impact analysis plot...")
    improved_trainer.plot_impact_analysis_robust('heloc_impact_analysis_robust.png')
    logger.info("✅ Robust impact analysis plot saved to heloc_impact_analysis_robust.png")
    
    # Save comprehensive results
    logger.info("💾 Saving comprehensive results...")
    results = {
        'training_cycles': len(training_history),
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'best_generator_loss': training_history[-1]['best_generator_loss'] if training_history else None,
        'training_history': training_history,
        'bert_fix_successful': improved_trainer.bert_fix_successful
    }
    
    with open('training_results_robust.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("✅ Comprehensive results saved to training_results_robust.json")
    
    logger.info("🎉 Robust improved discriminator training completed!")
    
    # Log final metrics to wandb
    if wandb.run is not None:
        if training_history:
            wandb.log({
                "final_best_generator_loss": training_history[-1]['best_generator_loss'],
                "training_cycles": len(training_history),
                "bert_fix_successful": improved_trainer.bert_fix_successful
            })
        wandb.finish()

if __name__ == "__main__":
    main()
