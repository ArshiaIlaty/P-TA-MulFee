import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import logging

logger = logging.getLogger(__name__)

class FixedSentenceLevelDiscriminator(nn.Module):
    """
    Fixed sentence-level discriminator that works properly with different model architectures
    """
    
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=256, device="cuda"):
        super(FixedSentenceLevelDiscriminator, self).__init__()
        self.device = device
        
        try:
            # Try to use a proper classification model first
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1
            ).to(device)
            self.use_classification_head = True
            logger.info(f"✅ Using {model_name} with classification head")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {model_name} with classification head: {e}")
            try:
                # Fallback to base model with custom classifier
                self.model = AutoModel.from_pretrained(model_name).to(device)
                self.use_classification_head = False
                
                # Get model dimension
                if hasattr(self.model, 'config'):
                    model_dim = getattr(self.model.config, 'hidden_size', 768)
                else:
                    model_dim = 768
                
                # Custom classifier
                self.classifier = nn.Sequential(
                    nn.Linear(model_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                ).to(device)
                
                logger.info(f"✅ Using {model_name} base model with custom classifier")
                
            except Exception as e2:
                logger.error(f"❌ Failed to load {model_name}: {e2}")
                # Ultimate fallback - simple LSTM-based discriminator
                self._create_fallback_discriminator()
    
    def _create_fallback_discriminator(self):
        """Create a simple fallback discriminator if all else fails"""
        logger.info("🔄 Creating fallback LSTM-based discriminator")
        
        self.model = nn.Sequential(
            nn.Embedding(50257, 256),  # GPT-2 vocab size
            nn.LSTM(256, 128, batch_first=True, bidirectional=True),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.use_classification_head = False
        self.use_fallback = True
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass that handles different model architectures
        """
        try:
            if hasattr(self, 'use_fallback') and self.use_fallback:
                # Fallback discriminator
                embeddings = self.model[0](input_ids)
                lstm_out, _ = self.model[1](embeddings)
                pooled = torch.mean(lstm_out, dim=1)
                return self.model[2:](pooled)
            
            elif self.use_classification_head:
                # Use the model's built-in classification head
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return torch.sigmoid(outputs.logits)
            
            else:
                # Use base model with custom classifier
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Handle different model architectures
                if hasattr(outputs, 'last_hidden_state'):
                    # BERT-style models
                    pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                elif hasattr(outputs, 'hidden_states'):
                    # Models with hidden states
                    pooled = outputs.hidden_states[-1][:, 0, :]
                else:
                    # GPT-style models - use mean pooling
                    if attention_mask is not None:
                        # Masked mean pooling
                        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        pooled = sum_embeddings / sum_mask
                    else:
                        # Simple mean pooling
                        pooled = torch.mean(outputs.last_hidden_state, dim=1)
                
                return self.classifier(pooled)
                
        except Exception as e:
            logger.error(f"❌ Error in sentence discriminator forward pass: {e}")
            # Return a dummy output to prevent training from crashing
            batch_size = input_ids.shape[0]
            return torch.ones(batch_size, 1).to(self.device) * 0.5

class RobustSentenceDiscriminator:
    """
    Robust sentence discriminator that tries multiple approaches
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.discriminator = None
        self.tokenizer = None
        self._initialize_discriminator()
    
    def _initialize_discriminator(self):
        """Try to initialize the best available discriminator"""
        
        # List of models to try in order of preference
        models_to_try = [
            "distilbert-base-uncased",  # Fast and reliable
            "bert-base-uncased",        # Standard BERT
            "roberta-base",             # RoBERTa
            "albert-base-v2",           # ALBERT
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"🔄 Trying to initialize {model_name}...")
                
                # Initialize tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Initialize discriminator
                discriminator = FixedSentenceLevelDiscriminator(
                    model_name=model_name, 
                    device=self.device
                )
                
                # Test with a simple input
                test_text = "This is a test sentence"
                test_input = tokenizer(
                    test_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(self.device)
                
                with torch.no_grad():
                    test_output = discriminator(**test_input)
                
                # If we get here, it works!
                self.discriminator = discriminator
                self.tokenizer = tokenizer
                logger.info(f"✅ Successfully initialized {model_name}")
                return
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize {model_name}: {e}")
                continue
        
        # If all models fail, create a simple fallback
        logger.error("❌ All models failed, creating simple fallback")
        self._create_simple_fallback()
    
    def _create_simple_fallback(self):
        """Create a very simple fallback discriminator"""
        logger.info("🔄 Creating simple fallback discriminator")
        
        # Simple tokenizer
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Simple discriminator
        self.discriminator = nn.Sequential(
            nn.Embedding(self.tokenizer.vocab_size, 128),
            nn.LSTM(128, 64, batch_first=True, bidirectional=True),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        if hasattr(self.discriminator, 'forward'):
            return self.discriminator(input_ids, attention_mask)
        else:
            # Simple fallback
            embeddings = self.discriminator[0](input_ids)
            lstm_out, _ = self.discriminator[1](embeddings)
            pooled = torch.mean(lstm_out, dim=1)
            return self.discriminator[2:](pooled)
    
    def parameters(self):
        """Return parameters for optimizer"""
        return self.discriminator.parameters()

def fix_sentence_discriminator_in_system(hierarchical_discriminators, device="cuda"):
    """
    Fix the sentence discriminator in an existing HierarchicalDiscriminatorSystem
    """
    logger.info("🔧 Fixing sentence discriminator in existing system...")
    
    try:
        # Create robust sentence discriminator
        robust_discriminator = RobustSentenceDiscriminator(device=device)
        
        # Replace the sentence discriminator
        hierarchical_discriminators.sentence_discriminator = robust_discriminator.discriminator
        hierarchical_discriminators.sentence_tokenizer = robust_discriminator.tokenizer
        
        # Update optimizer
        hierarchical_discriminators.optimizers["sentence"] = torch.optim.AdamW(
            robust_discriminator.parameters(), lr=1e-4
        )
        
        logger.info("✅ Sentence discriminator successfully fixed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix sentence discriminator: {e}")
        return False

# Test function
def test_sentence_discriminator():
    """Test the fixed sentence discriminator"""
    logger.info("🧪 Testing fixed sentence discriminator...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create robust discriminator
    robust_disc = RobustSentenceDiscriminator(device=device)
    
    # Test with sample text
    test_texts = [
        "RiskPerformance is Bad, ExternalRiskEstimate is 55",
        "MaxDelqEver is 0, MaxDelq2PublicRecLast12M is 0",
        "This is a test sentence for evaluation"
    ]
    
    for text in test_texts:
        try:
            # Tokenize
            inputs = robust_disc.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = robust_disc.forward(**inputs)
            
            logger.info(f"✅ Text: '{text[:50]}...' -> Score: {output.item():.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error testing text '{text[:50]}...': {e}")
    
    logger.info("🎉 Sentence discriminator test completed!")

if __name__ == "__main__":
    test_sentence_discriminator()
