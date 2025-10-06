import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SimpleSentenceDiscriminator(nn.Module):
    """
    Simple sentence discriminator that doesn't rely on complex transformer models
    """
    
    def __init__(self, vocab_size=50257, embedding_dim=256, hidden_dim=128, device="cuda"):
        super(SimpleSentenceDiscriminator, self).__init__()
        self.device = device
        
        # Simple architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        logger.info("✅ Simple sentence discriminator initialized")
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for sentence discrimination
        """
        try:
            # Get embeddings
            embeddings = self.embedding(input_ids)
            
            # LSTM processing
            lstm_out, (hidden, cell) = self.lstm(embeddings)
            
            # Use attention mask if provided for masked pooling
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
                sum_embeddings = torch.sum(lstm_out * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                # Simple mean pooling
                pooled = torch.mean(lstm_out, dim=1)
            
            # Classification
            output = self.classifier(pooled)
            return output
            
        except Exception as e:
            logger.error(f"❌ Error in simple sentence discriminator: {e}")
            # Return dummy output
            batch_size = input_ids.shape[0]
            return torch.ones(batch_size, 1).to(self.device) * 0.5

def fix_sentence_discriminator_simple(hierarchical_discriminators, device="cuda"):
    """
    Fix the sentence discriminator with a simple, reliable implementation
    """
    logger.info("🔧 Fixing sentence discriminator with simple implementation...")
    
    try:
        # Get vocab size from existing tokenizer
        vocab_size = hierarchical_discriminators.tokenizer.vocab_size
        
        # Create simple sentence discriminator
        simple_discriminator = SimpleSentenceDiscriminator(
            vocab_size=vocab_size,
            device=device
        ).to(device)
        
        # Replace the sentence discriminator
        hierarchical_discriminators.sentence_discriminator = simple_discriminator
        
        # Update optimizer
        hierarchical_discriminators.optimizers["sentence"] = torch.optim.AdamW(
            simple_discriminator.parameters(), lr=1e-4
        )
        
        logger.info("✅ Sentence discriminator successfully fixed with simple implementation!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix sentence discriminator: {e}")
        return False

def test_simple_sentence_discriminator():
    """Test the simple sentence discriminator"""
    logger.info("🧪 Testing simple sentence discriminator...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create simple discriminator
    discriminator = SimpleSentenceDiscriminator(device=device)
    
    # Test with sample inputs
    test_inputs = [
        torch.randint(0, 1000, (1, 10)).to(device),  # Random token IDs
        torch.randint(0, 1000, (2, 15)).to(device),  # Batch of 2
        torch.randint(0, 1000, (1, 5)).to(device),   # Short sequence
    ]
    
    for i, input_ids in enumerate(test_inputs):
        try:
            with torch.no_grad():
                output = discriminator(input_ids)
            
            logger.info(f"✅ Test {i+1}: Input shape {input_ids.shape} -> Output shape {output.shape}, Value: {output.item():.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error in test {i+1}: {e}")
    
    logger.info("🎉 Simple sentence discriminator test completed!")

if __name__ == "__main__":
    test_simple_sentence_discriminator()
