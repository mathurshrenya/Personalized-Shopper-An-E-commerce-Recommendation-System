import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pyspark.sql import SparkSession

def generate_synthetic_data(
    num_users: int = 1000,
    num_items: int = 500,
    num_interactions: int = 10000,
    output_path: str = "data/raw/interactions"
):
    """
    Generate synthetic user-item interaction data
    
    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        num_interactions: Number of interactions to generate
        output_path: Path to save the data
    """
    # Create Spark session
    spark = SparkSession.builder \
        .appName("Synthetic Data Generator") \
        .getOrCreate()
    
    # Generate random user and item IDs
    user_ids = [f"user_{i}" for i in range(num_users)]
    item_ids = [f"item_{i}" for i in range(num_items)]
    
    # Generate random interactions
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps within the last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    timestamps = [
        int((start_time + timedelta(
            seconds=np.random.randint(0, int((end_time - start_time).total_seconds()))
        )).timestamp())
        for _ in range(num_interactions)
    ]
    
    # Generate random ratings (1-5)
    ratings = np.random.randint(1, 6, num_interactions)
    
    # Create interaction data
    interactions = pd.DataFrame({
        'user_id': np.random.choice(user_ids, num_interactions),
        'item_id': np.random.choice(item_ids, num_interactions),
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Remove duplicate interactions
    interactions = interactions.drop_duplicates(['user_id', 'item_id', 'timestamp'])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to Spark DataFrame and save as Parquet
    spark_df = spark.createDataFrame(interactions)
    spark_df.write.format("parquet") \
        .mode("overwrite") \
        .save(output_path)
    
    print(f"Generated {len(interactions)} interactions")
    print(f"Data saved to {output_path}")
    
    # Generate and save user and item mappings
    user_map = {user: idx for idx, user in enumerate(user_ids)}
    item_map = {item: idx for idx, item in enumerate(item_ids)}
    
    # Save mappings
    os.makedirs("data/processed", exist_ok=True)
    
    with open("data/processed/user_map.json", "w") as f:
        import json
        json.dump(user_map, f)
    
    with open("data/processed/item_map.json", "w") as f:
        import json
        json.dump(item_map, f)
    
    print("User and item mappings saved")

if __name__ == "__main__":
    generate_synthetic_data() 