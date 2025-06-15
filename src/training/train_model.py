import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pyspark.sql import SparkSession
from delta.tables import DeltaTable
import logging
from typing import Tuple, List
import os
from datetime import datetime

from src.models.ncf_model import NCF, NCFTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractionDataset(Dataset):
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

class ModelTrainer:
    def __init__(
        self,
        spark_session: SparkSession,
        model_dir: str = "models",
        batch_size: int = 1024,
        num_epochs: int = 10,
        learning_rate: float = 0.001
    ):
        self.spark = spark_session
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data from Delta table"""
        df = self.spark.read.format("delta").load(data_path)
        
        # Convert to pandas for easier processing
        pdf = df.toPandas()
        
        # Get unique users and items
        self.user_map = {user: idx for idx, user in enumerate(pdf['user_id'].unique())}
        self.item_map = {item: idx for idx, item in enumerate(pdf['item_id'].unique())}
        
        # Convert to indices
        user_ids = np.array([self.user_map[user] for user in pdf['user_id']])
        item_ids = np.array([self.item_map[item] for item in pdf['item_id']])
        ratings = pdf['rating'].values
        
        return user_ids, item_ids, ratings
    
    def create_data_loaders(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders"""
        # Shuffle data
        indices = np.random.permutation(len(ratings))
        user_ids = user_ids[indices]
        item_ids = item_ids[indices]
        ratings = ratings[indices]
        
        # Split data
        train_size = int(len(ratings) * train_ratio)
        val_size = int(len(ratings) * val_ratio)
        
        train_dataset = InteractionDataset(
            user_ids[:train_size],
            item_ids[:train_size],
            ratings[:train_size]
        )
        
        val_dataset = InteractionDataset(
            user_ids[train_size:train_size + val_size],
            item_ids[train_size:train_size + val_size],
            ratings[train_size:train_size + val_size]
        )
        
        test_dataset = InteractionDataset(
            user_ids[train_size + val_size:],
            item_ids[train_size + val_size:],
            ratings[train_size + val_size:]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_users: int,
        num_items: int
    ):
        """Train the NCF model"""
        # Initialize model and trainer
        model = NCF(num_users, num_items)
        trainer = NCFTrainer(model, learning_rate=self.learning_rate)
        
        best_val_loss = float('inf')
        best_model_path = None
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss = 0
            for user_ids, item_ids, ratings in train_loader:
                loss = trainer.train_step(user_ids, item_ids, ratings)
                train_loss += loss
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0
            val_acc = 0
            for user_ids, item_ids, ratings in val_loader:
                loss, acc = trainer.evaluate(user_ids, item_ids, ratings)
                val_loss += loss
                val_acc += acc
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = os.path.join(self.model_dir, f"ncf_model_{timestamp}.pt")
                trainer.save_model(best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        
        return best_model_path
    
    def evaluate_model(self, model_path: str, test_loader: DataLoader):
        """Evaluate the trained model on test data"""
        # Load model
        model = NCF(len(self.user_map), len(self.item_map))
        trainer = NCFTrainer(model)
        trainer.load_model(model_path)
        
        # Evaluate
        test_loss = 0
        test_acc = 0
        for user_ids, item_ids, ratings in test_loader:
            loss, acc = trainer.evaluate(user_ids, item_ids, ratings)
            test_loss += loss
            test_acc += acc
        
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        
        logger.info(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
        
        return test_loss, test_acc

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("NCF Training") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Initialize trainer
    trainer = ModelTrainer(spark)
    
    # Load data
    user_ids, item_ids, ratings = trainer.load_data("data/processed/user_item_matrix")
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        user_ids, item_ids, ratings
    )
    
    # Train model
    best_model_path = trainer.train(
        train_loader,
        val_loader,
        len(trainer.user_map),
        len(trainer.item_map)
    )
    
    # Evaluate model
    trainer.evaluate_model(best_model_path, test_loader)

if __name__ == "__main__":
    main() 