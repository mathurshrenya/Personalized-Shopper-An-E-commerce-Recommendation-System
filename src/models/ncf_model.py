import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class NCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        layers: List[int] = [64, 32, 16, 8],
        dropout: float = 0.2
    ):
        """
        Neural Collaborative Filtering model
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Dimension of user and item embeddings
            layers: List of layer sizes for the MLP
            dropout: Dropout rate
        """
        super(NCF, self).__init__()
        
        # User and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_size = embedding_dim * 2
        
        for layer_size in layers:
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = layer_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            user_input: User indices tensor
            item_input: Item indices tensor
            
        Returns:
            Predicted ratings
        """
        # Get embeddings
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        
        # Concatenate embeddings
        concat = torch.cat([user_embedding, item_embedding], dim=1)
        
        # Pass through MLP layers
        x = concat
        for layer in self.mlp_layers:
            x = layer(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return torch.sigmoid(output)

class NCFTrainer:
    def __init__(
        self,
        model: NCF,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Trainer for the NCF model
        
        Args:
            model: NCF model instance
            learning_rate: Learning rate for optimization
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def train_step(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
        ratings: torch.Tensor
    ) -> float:
        """
        Perform a single training step
        
        Args:
            user_input: User indices tensor
            item_input: Item indices tensor
            ratings: Ground truth ratings
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move tensors to device
        user_input = user_input.to(self.device)
        item_input = item_input.to(self.device)
        ratings = ratings.to(self.device)
        
        # Forward pass
        predictions = self.model(user_input, item_input)
        loss = self.criterion(predictions, ratings)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
        ratings: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Evaluate the model
        
        Args:
            user_input: User indices tensor
            item_input: Item indices tensor
            ratings: Ground truth ratings
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        
        with torch.no_grad():
            user_input = user_input.to(self.device)
            item_input = item_input.to(self.device)
            ratings = ratings.to(self.device)
            
            predictions = self.model(user_input, item_input)
            loss = self.criterion(predictions, ratings)
            
            # Calculate accuracy
            predicted_ratings = (predictions > 0.5).float()
            accuracy = (predicted_ratings == ratings).float().mean()
            
        return loss.item(), accuracy.item()
    
    def save_model(self, path: str):
        """Save the model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 