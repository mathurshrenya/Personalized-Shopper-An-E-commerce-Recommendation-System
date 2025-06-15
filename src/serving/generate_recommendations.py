import torch
import numpy as np
from pyspark.sql import SparkSession
from delta.tables import DeltaTable
import logging
from typing import List, Dict, Tuple
import os
import json
from datetime import datetime

from src.models.ncf_model import NCF, NCFTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationGenerator:
    def __init__(
        self,
        spark_session: SparkSession,
        model_path: str,
        user_map_path: str,
        item_map_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the recommendation generator
        
        Args:
            spark_session: Spark session
            model_path: Path to the trained model
            user_map_path: Path to the user mapping file
            item_map_path: Path to the item mapping file
            device: Device to run inference on
        """
        self.spark = spark_session
        self.device = device
        
        # Load mappings
        with open(user_map_path, 'r') as f:
            self.user_map = json.load(f)
        with open(item_map_path, 'r') as f:
            self.item_map = json.load(f)
        
        # Create reverse mappings
        self.idx_to_user = {idx: user for user, idx in self.user_map.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_map.items()}
        
        # Load model
        self.model = NCF(len(self.user_map), len(self.item_map))
        self.trainer = NCFTrainer(self.model)
        self.trainer.load_model(model_path)
        self.model.eval()
        
        logger.info("Recommendation generator initialized")
    
    def get_user_recommendations(
        self,
        user_id: str,
        k: int = 10,
        exclude_items: List[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Generate top-k recommendations for a user
        
        Args:
            user_id: User ID
            k: Number of recommendations to generate
            exclude_items: List of items to exclude from recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_map:
            logger.warning(f"User {user_id} not found in mapping")
            return []
        
        # Convert user_id to index
        user_idx = self.user_map[user_id]
        
        # Create tensor for all items
        user_tensor = torch.LongTensor([user_idx] * len(self.item_map)).to(self.device)
        item_indices = torch.LongTensor(list(range(len(self.item_map)))).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(user_tensor, item_indices)
            predictions = predictions.cpu().numpy().flatten()
        
        # Create item-score pairs
        item_scores = [
            (self.idx_to_item[idx], float(score))
            for idx, score in enumerate(predictions)
        ]
        
        # Exclude items if specified
        if exclude_items:
            item_scores = [
                (item, score) for item, score in item_scores
                if item not in exclude_items
            ]
        
        # Sort by score and get top-k
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:k]
    
    def get_batch_recommendations(
        self,
        user_ids: List[str],
        k: int = 10,
        exclude_items: Dict[str, List[str]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users
        
        Args:
            user_ids: List of user IDs
            k: Number of recommendations per user
            exclude_items: Dictionary mapping user IDs to lists of items to exclude
            
        Returns:
            Dictionary mapping user IDs to lists of (item_id, score) tuples
        """
        recommendations = {}
        
        for user_id in user_ids:
            user_exclude_items = exclude_items.get(user_id, []) if exclude_items else None
            recommendations[user_id] = self.get_user_recommendations(
                user_id,
                k=k,
                exclude_items=user_exclude_items
            )
        
        return recommendations
    
    def save_recommendations(
        self,
        recommendations: Dict[str, List[Tuple[str, float]]],
        output_path: str
    ):
        """Save recommendations to a Delta table"""
        # Convert recommendations to DataFrame format
        rows = []
        for user_id, items in recommendations.items():
            for item_id, score in items:
                rows.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'score': score,
                    'timestamp': datetime.now()
                })
        
        # Create DataFrame and save to Delta
        df = self.spark.createDataFrame(rows)
        df.write.format("delta") \
            .mode("overwrite") \
            .partitionBy("user_id") \
            .save(output_path)
        
        logger.info(f"Saved recommendations to {output_path}")

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Recommendation Generator") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Initialize recommendation generator
    generator = RecommendationGenerator(
        spark,
        model_path="models/ncf_model_latest.pt",
        user_map_path="data/processed/user_map.json",
        item_map_path="data/processed/item_map.json"
    )
    
    # Load users to generate recommendations for
    users_df = spark.read.format("delta").load("data/processed/users")
    user_ids = [row.user_id for row in users_df.collect()]
    
    # Generate recommendations
    recommendations = generator.get_batch_recommendations(
        user_ids,
        k=10
    )
    
    # Save recommendations
    generator.save_recommendations(
        recommendations,
        "data/recommendations/latest"
    )

if __name__ == "__main__":
    main() 