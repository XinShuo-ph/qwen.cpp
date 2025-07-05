import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

from gpt2_model import GPT2Model

class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration purposes"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Simple character-level tokenization for demo
        tokens = [ord(c) for c in text[:self.max_length]]
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)

class GPT2Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device='cuda'):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_dataloader) * 10
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_dataloader, desc="Training"):
            batch = batch.to(self.device)
            
            # Prepare inputs and targets
            inputs = batch[:, :-1]  # All tokens except the last
            targets = batch[:, 1:]  # All tokens except the first
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_dataloader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                batch = batch.to(self.device)
                
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                logits = self.model(inputs)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)
    
    def train(self, num_epochs=10):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

def create_sample_data():
    """Create sample text data for demonstration"""
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Transformers revolutionized the field of NLP with attention mechanisms.",
        "GPT models are autoregressive language models based on the transformer architecture.",
        "Python is a popular programming language for machine learning.",
        "PyTorch is a deep learning framework developed by Facebook.",
        "Neural networks are inspired by the human brain's structure.",
        "Artificial intelligence aims to create intelligent machines.",
    ] * 100  # Repeat for more training data
    
    return samples

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    texts = create_sample_data()
    
    # Create datasets
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    train_dataset = SimpleTextDataset(train_texts, None, max_length=128)
    val_dataset = SimpleTextDataset(val_texts, None, max_length=128)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = GPT2Model(
        vocab_size=256,  # ASCII characters
        d_model=256,
        num_heads=8,
        num_layers=4,
        max_seq_length=128,
        d_ff=1024
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = GPT2Trainer(model, train_loader, val_loader, device)
    
    # Train the model
    trainer.train(num_epochs=5)
    
    # Save final model
    torch.save(model.state_dict(), "gpt2_model_final.pt")
    print("Final model saved!")

if __name__ == "__main__":
    main()