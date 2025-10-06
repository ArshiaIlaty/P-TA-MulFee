import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from utils import format_row

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _sanity_check_ids(ids, embedding):
    """Sanity check for token IDs to prevent out-of-range errors"""
    vocab = embedding.num_embeddings
    if ids.dtype != torch.long:
        raise RuntimeError(f"input_ids dtype {ids.dtype}, expected torch.long")
    bad = (ids < 0) | (ids >= vocab)
    if bad.any():
        bad_vals = ids[bad]
        raise RuntimeError(f"Out-of-range token ids detected. min={int(bad_vals.min())}, "
                           f"max={int(bad_vals.max())}, vocab={vocab}")


class FeatureWiseDiscriminator(nn.Module):
    """
    Feature-wise discriminator that evaluates plausibility of specific feature combinations
    """

    def __init__(self, feature_name, input_dim=768, hidden_dim=256):
        super(FeatureWiseDiscriminator, self).__init__()
        self.feature_name = feature_name
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


class TokenLevelDiscriminator(nn.Module):
    """
    Token-level discriminator for evaluating individual token plausibility
    """

    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256):
        super(TokenLevelDiscriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embeddings)
        # Average pooling over sequence length
        pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled)


class SentenceLevelDiscriminator(nn.Module):
    """
    Sentence-level discriminator for evaluating complete feature-value pairs
    """

    def __init__(self, model_name="gpt2", hidden_dim=256):
        super(SentenceLevelDiscriminator, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, output_hidden_states=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.hidden_states[-1][:, 0, :]
        return self.classifier(cls_output)


class RowLevelDiscriminator(nn.Module):
    """
    Row-level discriminator for evaluating complete data rows
    """

    def __init__(self, model_name="gpt2", hidden_dim=256):
        super(RowLevelDiscriminator, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, output_hidden_states=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.hidden_states[-1][:, 0, :]
        return self.classifier(cls_output)


class HierarchicalDiscriminatorSystem:
    """
    Complete hierarchical discriminator system with multi-level feedback
    """

    def __init__(self, model_name="gpt2", device="cuda", dataset_type="generic"):
        self.device = device
        self.dataset_type = dataset_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize discriminators
        self.token_discriminator = TokenLevelDiscriminator(
            vocab_size=self.tokenizer.vocab_size
        ).to(device)

        self.sentence_discriminator = SentenceLevelDiscriminator(
            model_name=model_name
        ).to(device)

        self.row_discriminator = RowLevelDiscriminator(model_name=model_name).to(device)

        # Initialize a shared embedding model for feature extraction (only once)
        self.embedding_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device)
        self.embedding_model.eval()  # Set to evaluation mode

        # Feature-wise discriminators - now dataset-agnostic
        self.feature_discriminators = {}
        
        # Initialize feature discriminators based on dataset type
        if dataset_type == "diabetes":
            self.feature_discriminators = {
                "age_bmi": FeatureWiseDiscriminator("age_bmi").to(device),
                "glucose_hba1c": FeatureWiseDiscriminator("glucose_hba1c").to(device),
                "hypertension_heart": FeatureWiseDiscriminator("hypertension_heart").to(device),
                "smoking_diabetes": FeatureWiseDiscriminator("smoking_diabetes").to(device),
            }
        elif dataset_type == "heloc":
            self.feature_discriminators = {
                "risk_estimate": FeatureWiseDiscriminator("risk_estimate").to(device),
                "trade_history": FeatureWiseDiscriminator("trade_history").to(device),
                "delinquency": FeatureWiseDiscriminator("delinquency").to(device),
                "credit_utilization": FeatureWiseDiscriminator("credit_utilization").to(device),
            }
        else:
            # Generic discriminators for any dataset
            self.feature_discriminators = {
                "feature_group_1": FeatureWiseDiscriminator("feature_group_1").to(device),
                "feature_group_2": FeatureWiseDiscriminator("feature_group_2").to(device),
                "feature_group_3": FeatureWiseDiscriminator("feature_group_3").to(device),
                "feature_group_4": FeatureWiseDiscriminator("feature_group_4").to(device),
            }

        # Optimizers for each discriminator with improved learning rates
        discriminator_lr = 5e-4  # Increased learning rate for better training
        self.optimizers = {
            "token": torch.optim.AdamW(self.token_discriminator.parameters(), lr=discriminator_lr),
            "sentence": torch.optim.AdamW(
                self.sentence_discriminator.parameters(), lr=discriminator_lr
            ),
            "row": torch.optim.AdamW(self.row_discriminator.parameters(), lr=discriminator_lr),
            "features": {
                name: torch.optim.AdamW(disc.parameters(), lr=discriminator_lr)
                for name, disc in self.feature_discriminators.items()
            },
        }

    def extract_feature_embeddings(self, text, feature_name):
        """
        Extract embeddings for specific feature combinations using shared model
        """
        # Tokenize the text
        encoding = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)

        # Use the shared embedding model
        with torch.no_grad():
            outputs = self.embedding_model(**encoding, output_hidden_states=True)
            # Use last hidden state
            embeddings = outputs.hidden_states[-1].mean(dim=1)  # Average pooling

        return embeddings

    def get_multi_level_feedback(self, generated_text, real_text=None):
        """
        Get feedback from all discriminators at different levels
        """
        feedback = {}

        # Token-level feedback
        token_ids = self.tokenizer.encode(generated_text, return_tensors="pt").to(
            self.device
        )
        token_feedback = self.token_discriminator(token_ids)
        feedback["token"] = token_feedback.item()

        # Sentence-level feedback (for each feature-value pair)
        sentences = generated_text.split(", ")
        sentence_feedbacks = []
        for sentence in sentences:
            if " is " in sentence:
                # Use sentence_tokenizer if available, otherwise fall back to main tokenizer
                tokenizer_to_use = getattr(self, 'sentence_tokenizer', self.tokenizer)
                encoding = tokenizer_to_use(
                    sentence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                ).to(self.device)
                
                # Sanity check for token IDs
                if hasattr(self.sentence_discriminator, 'get_input_embeddings'):
                    try:
                        _sanity_check_ids(encoding["input_ids"], self.sentence_discriminator.get_input_embeddings())
                    except RuntimeError as e:
                        logger.warning(f"Sentence discriminator sanity check failed: {e}")
                        continue
                
                with torch.no_grad():
                    if hasattr(self.sentence_discriminator, 'classifier'):
                        # Fixed sentence discriminator (returns probabilities)
                        sentence_feedback = self.sentence_discriminator(
                            encoding["input_ids"], encoding["attention_mask"]
                        )
                    elif hasattr(self.sentence_discriminator, 'logits'):
                        # DistilBERT style
                        out = self.sentence_discriminator(**encoding)
                        sentence_feedback = out.logits.squeeze()
                    else:
                        # GPT-2 style
                        sentence_feedback = self.sentence_discriminator(
                            encoding["input_ids"], encoding["attention_mask"]
                        )
                        # Handle case where it returns a SequenceClassifierOutput
                        if hasattr(sentence_feedback, 'logits'):
                            sentence_feedback = sentence_feedback.logits.squeeze()
                sentence_feedbacks.append(sentence_feedback.item())

        feedback["sentence"] = (
            np.mean(sentence_feedbacks) if sentence_feedbacks else 0.5
        )

        # Row-level feedback
        row_encoding = self.tokenizer(
            generated_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        row_feedback = self.row_discriminator(
            row_encoding["input_ids"], row_encoding["attention_mask"]
        )
        feedback["row"] = row_feedback.item()

        # Feature-wise feedback
        feature_feedbacks = {}
        for feature_name, discriminator in self.feature_discriminators.items():
            embeddings = self.extract_feature_embeddings(generated_text, feature_name)
            feature_feedback = discriminator(embeddings)
            feature_feedbacks[feature_name] = feature_feedback.item()

        feedback["features"] = feature_feedbacks

        return feedback

    def train_discriminators(self, real_texts, synthetic_texts, epochs=3):
        """
        Train all discriminators on real vs synthetic data
        """
        logger.info("Training hierarchical discriminators...")

        # Prepare data
        real_labels = torch.ones(len(real_texts)).to(self.device)
        synthetic_labels = torch.zeros(len(synthetic_texts)).to(self.device)

        all_texts = real_texts + synthetic_texts
        all_labels = torch.cat([real_labels, synthetic_labels])

        # Token-level training
        logger.info("Training token-level discriminator...")
        for epoch in range(epochs):
            total_loss = 0
            for i, text in enumerate(all_texts):
                token_ids = self.tokenizer.encode(text, return_tensors="pt").to(
                    self.device
                )
                label = all_labels[i].unsqueeze(0).float()  # Ensure float type

                self.optimizers["token"].zero_grad()
                prediction = self.token_discriminator(token_ids)
                # Ensure prediction has the same shape as label
                if prediction.shape != label.shape:
                    prediction = prediction.squeeze()
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)
                loss = F.binary_cross_entropy(prediction, label)
                loss.backward()
                self.optimizers["token"].step()
                total_loss += loss.item()

            logger.info(
                f"Token discriminator epoch {epoch+1}, avg loss: {total_loss/len(all_texts):.4f}"
            )

        # Sentence-level training
        logger.info("Training sentence-level discriminator...")
        for epoch in range(epochs):
            total_loss = 0
            for i, text in enumerate(all_texts):
                sentences = text.split(", ")
                for sentence in sentences:
                    if " is " in sentence:
                        # Use sentence_tokenizer if available, otherwise fall back to main tokenizer
                        tokenizer_to_use = getattr(self, 'sentence_tokenizer', self.tokenizer)
                        encoding = tokenizer_to_use(
                            sentence,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=64,
                        ).to(self.device)
                        label = all_labels[i].unsqueeze(0).float()  # Ensure float type

                        # Sanity check for token IDs
                        if hasattr(self.sentence_discriminator, 'get_input_embeddings'):
                            try:
                                _sanity_check_ids(encoding["input_ids"], self.sentence_discriminator.get_input_embeddings())
                            except RuntimeError as e:
                                logger.warning(f"Sentence discriminator training sanity check failed: {e}")
                                continue

                        self.optimizers["sentence"].zero_grad()
                        
                        # Handle different model types
                        if hasattr(self.sentence_discriminator, 'classifier'):
                            # Fixed sentence discriminator (returns probabilities)
                            prediction = self.sentence_discriminator(
                                encoding["input_ids"], encoding["attention_mask"]
                            )
                        elif hasattr(self.sentence_discriminator, 'logits'):
                            # DistilBERT style
                            out = self.sentence_discriminator(**encoding)
                            prediction = out.logits.squeeze()
                        else:
                            # GPT-2 style
                            prediction = self.sentence_discriminator(
                                encoding["input_ids"], encoding["attention_mask"]
                            )
                            # Handle case where it returns a SequenceClassifierOutput
                            if hasattr(prediction, 'logits'):
                                prediction = prediction.logits.squeeze()
                        
                        # Ensure prediction has the same shape as label
                        if prediction.shape != label.shape:
                            prediction = prediction.squeeze()
                            if prediction.dim() == 0:
                                prediction = prediction.unsqueeze(0)
                        
                        # Use appropriate loss function based on discriminator type
                        if hasattr(self.sentence_discriminator, 'classifier'):
                            # Fixed discriminator outputs probabilities, use BCE
                            loss = F.binary_cross_entropy(prediction, label)
                        else:
                            # Original discriminator outputs logits, use BCE with logits
                            loss = F.binary_cross_entropy_with_logits(prediction, label)
                        loss.backward()
                        self.optimizers["sentence"].step()
                        total_loss += loss.item()

            logger.info(
                f"Sentence discriminator epoch {epoch+1}, avg loss: {total_loss/len(all_texts):.4f}"
            )

        # Row-level training
        logger.info("Training row-level discriminator...")
        for epoch in range(epochs):
            total_loss = 0
            for i, text in enumerate(all_texts):
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)
                label = all_labels[i].unsqueeze(0).float()  # Ensure float type

                self.optimizers["row"].zero_grad()
                prediction = self.row_discriminator(
                    encoding["input_ids"], encoding["attention_mask"]
                )
                # Ensure prediction has the same shape as label
                if prediction.shape != label.shape:
                    prediction = prediction.squeeze()
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)
                loss = F.binary_cross_entropy(prediction, label)
                loss.backward()
                self.optimizers["row"].step()
                total_loss += loss.item()

            logger.info(
                f"Row discriminator epoch {epoch+1}, avg loss: {total_loss/len(all_texts):.4f}"
            )

    def save_discriminators(self, save_path="./hierarchical_discriminators"):
        """
        Save all discriminators
        """
        torch.save(
            {
                "token_discriminator": self.token_discriminator.state_dict(),
                "sentence_discriminator": self.sentence_discriminator.state_dict(),
                "row_discriminator": self.row_discriminator.state_dict(),
                "feature_discriminators": {
                    name: disc.state_dict()
                    for name, disc in self.feature_discriminators.items()
                },
            },
            f"{save_path}.pth",
        )
        logger.info(f"Discriminators saved to {save_path}.pth")

    def load_discriminators(self, load_path="./hierarchical_discriminators"):
        """
        Load all discriminators
        """
        checkpoint = torch.load(f"{load_path}.pth")
        self.token_discriminator.load_state_dict(checkpoint["token_discriminator"])
        self.sentence_discriminator.load_state_dict(
            checkpoint["sentence_discriminator"]
        )
        self.row_discriminator.load_state_dict(checkpoint["row_discriminator"])
        for name, disc in self.feature_discriminators.items():
            disc.load_state_dict(checkpoint["feature_discriminators"][name])
        logger.info(f"Discriminators loaded from {load_path}.pth")


def integrate_with_great_training(
    great_model,
    tokenizer,
    hierarchical_discriminators,
    real_data,
    synthetic_data,
    device="cuda",
):
    """
    Integrate hierarchical discriminators with GREAT training
    """
    logger.info("Integrating hierarchical discriminators with GREAT training...")

    # Train discriminators first
    hierarchical_discriminators.train_discriminators(real_data, synthetic_data)

    # Use discriminators to guide GREAT training
    great_model.train()
    optimizer = torch.optim.AdamW(great_model.parameters(), lr=5e-5)

    for epoch in range(3):
        total_loss = 0
        for i, real_text in enumerate(real_data):
            # Generate synthetic text using GREAT
            input_ids = tokenizer.encode(real_text, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = great_model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 20,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                )

            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # Get multi-level feedback
            feedback = hierarchical_discriminators.get_multi_level_feedback(
                generated_text
            )

            # Calculate weighted loss based on feedback
            token_weight = 0.2
            sentence_weight = 0.3
            row_weight = 0.3
            feature_weight = 0.2

            # Convert feedback to loss (lower feedback = higher loss)
            token_loss = 1 - feedback["token"]
            sentence_loss = 1 - feedback["sentence"]
            row_loss = 1 - feedback["row"]
            feature_loss = 1 - np.mean(list(feedback["features"].values()))

            total_feedback_loss = (
                token_weight * token_loss
                + sentence_weight * sentence_loss
                + row_weight * row_loss
                + feature_weight * feature_loss
            )

            # Standard language modeling loss
            outputs = great_model(input_ids=input_ids, labels=input_ids)
            lm_loss = outputs.loss

            # Combined loss
            combined_loss = lm_loss + 0.1 * total_feedback_loss

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            total_loss += combined_loss.item()

            if i % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {i}, Loss: {combined_loss.item():.4f}"
                )

        avg_loss = total_loss / len(real_data)
        logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize hierarchical discriminator system
    hierarchical_discriminators = HierarchicalDiscriminatorSystem(device=device)

    # Load diabetes data
    df = pd.read_csv("diabetes.csv")
    real_texts = df.apply(format_row, axis=1).tolist()

    # Generate some synthetic texts for training (you would use your GREAT model here)
    synthetic_texts = real_texts[
        :100
    ]  # Placeholder - replace with actual synthetic data

    # Train discriminators
    hierarchical_discriminators.train_discriminators(real_texts, synthetic_texts)

    # Save discriminators
    hierarchical_discriminators.save_discriminators()

    logger.info("Hierarchical discriminator system ready for integration!")
